import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import *

def get_config(model_path, num_classes, percent, ancestor=False):
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if ancestor:
        model = densenet.densenet121(weights=None)
    else:
        cfg = checkpoint['cfg']
        model = densenet.densenet121(weights=None, cfg=cfg)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['state_dict'])

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
    thre_index = int(total * percent)  # The index of the threshold
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []

    for layer_id in range(len(old_modules)):
        m = old_modules[layer_id]
        if isinstance(m, nn.BatchNorm2d) and isinstance(old_modules[layer_id - 1], nn.Conv2d) and not isinstance(old_modules[layer_id + 1], nn.Linear):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

    return cfg, cfg_mask

def remove_zero(a):
    mask = (a != 0)
    return torch.masked_select(a, mask)

def index_to_mask(index):
    mask = torch.zeros(128)
    mask[index] = 1

    return mask

def load_state_from_original_network(synthesis_model, num_classes, fold_num, percent, dataset):
    ancestor_path = 'trained_models/penalty_0/ancestor/{}/Fold_'+str(fold_num)+'/pruned_checkpoint_0.pth.tar'
    model_path = 'trained_models/penalty_0/trained_models/ratio_0.1/{}/Fold_'+str(fold_num)+'/pruned_checkpoint_{}.pth.tar'
    cfgs, cfg_masks = [], []

    cfg, cfg_mask = get_config(ancestor_path.format(dataset), num_classes, percent, ancestor=True)
    cfgs.append(cfg)
    cfg_masks.append(cfg_mask)

    for gen in range(1, 10):
        cfg, cfg_mask = get_config(model_path.format(dataset, gen), num_classes, percent, ancestor=False)
        cfgs.append(cfg)
        cfg_masks.append(cfg_mask)

    index_maps = []
    for i in range(58):
        index = torch.range(1, 128)
        for j in range(10):
            index = index.mul_(cfg_masks[j][i])
            index = remove_zero(index)
        index_maps.append(index)
    
    # The original model fined tuned on the medical image dataset
    checkpoint = torch.load(ancestor_path.format(dataset))
    original_model = densenet.densenet121(weights=None)
    num_ftrs = original_model.classifier.in_features
    original_model.classifier = nn.Linear(num_ftrs, num_classes)
    original_model.load_state_dict(checkpoint['state_dict'])

    old_modules = list(original_model.modules())
    new_modules = list(synthesis_model.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(64)
    end_mask = index_to_mask(index_maps[layer_id_in_cfg].type(torch.LongTensor).clone() - 1)
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
                if layer_id_in_cfg < len(index_maps):
                    # end_mask = cfg_mask[layer_id_in_cfg]
                    end_mask = index_to_mask(index_maps[layer_id_in_cfg].type(torch.LongTensor).clone() - 1)
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