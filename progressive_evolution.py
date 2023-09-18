import os
import argparse

# python progressive_evolution.py --percent 3 --num_classes 3 --dataset sarscov2-ctscan --data_file sars-cov2-ctscan --s 0.0001

# Training settings
parser = argparse.ArgumentParser(description='Progressive evolution setting')
parser.add_argument('--percent', type=float, default=0.1,
                      help='The percentage of scaling factors to be remove')
parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes to classify')
parser.add_argument('--dataset', type=str, default='brain-tumor',
                    help='training dataset (default: brain-tumor)')
parser.add_argument('--data_file', type=str, default='brain-tumor',
                      help='The file name of dataset')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--generation', type=int, default=10,
                    help='the total number of generations (default: 10)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--fold', default=5, type=int,
                    help='The total number of fold in cross-validation')

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

args = parser.parse_args()

# For pruning
str_python = 'python -W ignore {} '
str_percent = '--percent {} '.format(args.percent)
str_model_a = '--model trained_models/penalty_{}/ancestor/{}/Fold_{}/pruned_checkpoint_0.pth.tar '
str_model_d = '--model trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}/pruned_checkpoint_{}.pth.tar '
str_save_a = '--save trained_models/penalty_{}/ancestor/{}/Fold_{}/ '
str_save_d = '--save trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}/ '
str_prune = '--save trained_models/penalty_{}/intermediate/ratio_{}/{}/Fold_{}/ '
str_class = '--num_classes {} '.format(args.num_classes)
str_generation = '--generation {} '

# For refining when penalty applied
if args.s == 0:
    str_sparsity = ''
    penalty = 0
else:
    str_sparsity = '-sr --s {} '.format(args.s)
    penalty = args.s

str_refine = '--refine trained_models/penalty_{}/intermediate/ratio_{}/{}/Fold_{}/pruned_{}.pth.tar '
str_epoch = '--epochs {} '.format(args.epochs)
str_data = '--data_dir ../data/mat/{}_Fold_{}.mat '
str_fold = '--fold_num {}'

str_cmd = ''

for fold_num in range(1, args.fold+1):
    ancestor_file = 'trained_models/penalty_{}/ancestor/{}/Fold_{}/pruned_checkpoint_0.pth.tar'.format(penalty, args.dataset, fold_num)
    if not os.path.isfile(ancestor_file):
        save_dir = 'trained_models/penalty_{}/ancestor/{}/Fold_{}/'.format(penalty, args.dataset, fold_num)
        check_folder(save_dir)

        print('Generate the ancestor network')
        str_cmd = str_python.format('main.py') + str_sparsity + str_epoch.format(args.epochs) + str_class.format(args.num_classes) + str_save_a.format(penalty, args.dataset, fold_num) + str_data.format(args.data_file, fold_num) + str_fold.format(fold_num)
        print(str_cmd)
        os.system(str_cmd)

for fold_num in range(1, args.fold+1):
    print('Runing on Fold {}'.format(fold_num))
    for generation in range(1, args.generation+1):
        # Pruning
        save_dir = 'trained_models/penalty_{}/intermediate/ratio_{}/{}/Fold_{}/'.format(penalty, args.percent, args.dataset, fold_num)
        check_folder(save_dir)

        print('Generating geneartion {}'.format(generation))
        if generation == 1:
            str_cmd = str_python.format('denseprune.py') + str_percent + str_model_a.format(penalty, args.dataset, fold_num) + str_prune.format(penalty, args.percent, args.dataset, fold_num) + str_class + str_generation.format(generation)
        else:
            str_cmd = str_python.format('denseprune.py') + str_percent + str_model_d.format(penalty, args.percent, args.dataset, fold_num, generation-1) + str_prune.format(penalty, args.percent, args.dataset, fold_num) + str_class + str_generation.format(generation)
        
        print(str_cmd)
        code = os.system(str_cmd)
        # Fail to generate a new descendant network, the current generation is the last generation under the ratio setting
        if code != 0:
            break

        # Refining
        save_dir = 'trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}/'.format(penalty, args.percent, args.dataset, fold_num)
        check_folder(save_dir)

        print('Refining geneartion {}'.format(generation))
        str_cmd = str_python.format('main.py') + str_sparsity + str_refine.format(penalty, args.percent, args.dataset, fold_num, generation) + str_save_d.format(penalty, args.percent, args.dataset, fold_num) + str_epoch + str_class + str_data.format(args.data_file, fold_num) + str_generation.format(generation) + str_fold.format(fold_num)
        
        print(str_cmd)
        os.system(str_cmd)