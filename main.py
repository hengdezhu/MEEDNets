from __future__ import print_function
import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import models

from tqdm import tqdm
import time

from data_loader import get_data_loader_from_mat
from filter_analysis import load_state_from_original_network
from utils import accuracy, AverageMeter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes to classify')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data_dir', type=str, default='./data/cifar100',
                      help='Directory in which data is stored')
parser.add_argument('--fold_num', type=str, default='1',
                      help='Specify which fold of data to be loaded')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--generation', type=int, default=0, metavar='G',
                    help='the generation (default: 0)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--knowledge', '-k', dest='knowledge', action='store_false',
                    help='not inherit weights')
parser.add_argument('--original', '-o', dest='original', action='store_true',
                    help='inherit weights from the very original model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

train_loader, test_loader = get_data_loader_from_mat(args.data_dir, args.batch_size, args.seed, fold_num=args.fold_num, **kwargs)

cfg = []

if args.refine:
    # checkpoint = torch.load('densenet/pruned_{}.pth.tar'.format(args.generation))
    checkpoint = torch.load(args.refine)
    cfg = checkpoint['cfg']
    model = models.densenet121(weights=None, cfg=cfg)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.num_classes)
    if args.knowledge:
        if args.original:
            print('Load pre-trained weights from the pre-trained original network')
            # copy the weights from the pre-trained original network to the pruned network
            load_state_from_original_network(model, args.num_classes, args.fold_num, args.percent, args.dataset)
        else:
            print('Load pre-trained weights from the pruned network...')
            model.load_state_dict(checkpoint['state_dict'])
else:
    if args.knowledge:
        print('Load pre-trained weights...')
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
    else:
        model = models.densenet121(weights=None)
    # model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.num_classes)

modules = list(model.modules())

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_ce = nn.CrossEntropyLoss()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for layer_id in range(len(modules)):
        m = modules[layer_id]
        # Only penalise the second BN in a DenseLayer
        if isinstance(m, nn.BatchNorm2d) and isinstance(modules[layer_id - 1], nn.Conv2d) and not isinstance(modules[layer_id + 1], nn.Linear):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))

def train(epoch):
    print(
        "Epoch: {}/{} - LR: {:.6F}".format(
            epoch+1, args.epochs, args.lr
        )
    )

    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    accies = AverageMeter()

    tic = time.time()

    with tqdm(total=len(train_loader.dataset)) as pbar:
        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.cuda:
                images,  labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            output = model(images)
            loss = loss_ce(output, labels)
            prec = accuracy(output.data, labels.data, topk=(1,))[0]

            losses.update(loss.item(), images.size()[0])
            accies.update(prec.item(), images.size()[0])

            optimizer.zero_grad()
            loss.backward()
            if args.sr:
                updateBN()
            optimizer.step()

            toc = time.time()
            batch_time.update(toc-tic)

            pbar.set_description(
                (
                    "{:.1f}s - training loss: {:.4f} - training acc: {:.4f}".format(
                        (toc-tic), losses.avg, accies.avg
                    )
                )
            )

            batch_size = images.shape[0]
            pbar.update(batch_size)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, filepath):
    checkpoint_file = 'pruned_checkpoint_{}.pth.tar'.format(args.generation) if args.refine else 'pruned_checkpoint_0.pth.tar'
    torch.save(state, os.path.join(filepath, checkpoint_file))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    # test()
    save_info = {
        'cfg':cfg,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(save_info, filepath=args.save)