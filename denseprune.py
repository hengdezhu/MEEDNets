import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes to classify')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--data_dir', type=str, default='./data/cifar100',
                    help='Directory in which data is stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=40,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--generation', type=int, default=0, metavar='G',
                    help='the generation (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        cfg = checkpoint['cfg']
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if len(cfg) == 0:
    model = densenet.densenet121(weights=None)
else:
    model = densenet.densenet121(weights=None, cfg=cfg)
    
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, args.num_classes)

if args.cuda:
    model.cuda()

model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}'".format(args.model))
        
old_num_parameters = sum([param.nelement() for param in model.parameters()])

old_modules = list(model.modules())

total = 0

for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and isinstance(old_modules[layer_id - 1], nn.Conv2d) and not isinstance(old_modules[layer_id + 1], nn.Linear):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0

for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and isinstance(old_modules[layer_id - 1], nn.Conv2d) and not isinstance(old_modules[layer_id + 1], nn.Linear):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)  # The index of the threshold
thre = y[thre_index]

# simple set BN scales and shifts to zeros
pruned = 0
cfg = []
cfg_mask = []

for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and isinstance(old_modules[layer_id - 1], nn.Conv2d) and not isinstance(old_modules[layer_id + 1], nn.Linear):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        # print(
        #     'layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
        #     format(layer_id, mask.shape[0], int(torch.sum(mask)))
        # )

pruned_ratio = pruned/total

print('Pre-processing Successful!')

print("Cfg:")
print(cfg)

newmodel = densenet.densenet121(weights=None, cfg=cfg)
num_ftrs = newmodel.classifier.in_features
newmodel.classifier = nn.Linear(num_ftrs, args.num_classes)

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune_{}.txt".format(args.generation))
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Previous number of parameters: \n"+str(old_num_parameters)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")

new_modules = list(newmodel.modules())

layer_id_in_cfg = 0
start_mask = torch.ones(64)
end_mask = cfg_mask[layer_id_in_cfg]
first_conv = True

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]

    if isinstance(m0, nn.BatchNorm2d):
        if isinstance(old_modules[layer_id - 1], nn.Conv2d) and not isinstance(old_modules[layer_id + 1], nn.Linear):
            # if the previous layer is a conv layer and the following layer is not nn.Linear, then the current batch normalization layer will be pruned.
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            
            m1.weight.data = m0.weight.data[idx1.tolist()]
            m1.bias.data = m0.bias.data[idx1.tolist()]

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            # for the rest of BN layer, they will not be pruned
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

        continue

    elif isinstance(m0, nn.Conv2d):
        if first_conv:
            # We don't change the first convolution layer.
            m1.weight.data = m0.weight.data.clone()
            first_conv = False
            continue
        if isinstance(old_modules[layer_id + 1], nn.AvgPool2d):
            # We don't change the convolution layer in the transition layer
            m1.weight.data = m0.weight.data.clone()
            continue
        if isinstance(old_modules[layer_id + 1], nn.BatchNorm2d) and not isinstance(old_modules[layer_id - 5], nn.BatchNorm2d):
            # We reduce the number of filters
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue
        if isinstance(old_modules[layer_id - 5], nn.BatchNorm2d):
            # We reduce the channel of each filters
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            continue
    
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned_{}.pth.tar'.format(args.generation)))

# print(newmodel)