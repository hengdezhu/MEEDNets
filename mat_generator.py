import numpy as np

import random

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset

from sklearn.model_selection import KFold
import scipy.io as sio

def get_dataset(data_dir):
    trans = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=trans)

    return dataset

dataset = get_dataset('../data/covid-pneumonia-normal-chest-xray-images/')
print(dataset.class_to_idx)

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print('-------------------------------')
    print(f'FOLD {fold+1}')
    
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    train_set = Subset(dataset, train_ids)
    test_set = Subset(dataset, test_ids)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=False)
    
    print('Processing Training Data')
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        train_data.append(images.numpy())
        train_label.append(labels.numpy())
        
    train_data = (*train_data,)
    train_data = np.concatenate(train_data, axis=0)
    train_label = (*train_label,)
    train_label = np.concatenate(train_label, axis=0)
    
    print('Processing Test Data')
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        test_data.append(images.numpy())
        test_label.append(labels.numpy())
        
    test_data = (*test_data,)
    test_data = np.concatenate(test_data, axis=0)
    test_label = (*test_label,)
    test_label = np.concatenate(test_label, axis=0)
    
    sio.savemat('../data/mat/xray_dataset_covid19_Fold_{}.mat'.format(fold+1), {
        'train_data_Fold_{}'.format(fold+1):train_data,
        'train_label_Fold_{}'.format(fold+1):train_label,
        'test_data_Fold_{}'.format(fold+1):test_data,
        'test_label_Fold_{}'.format(fold+1):test_label,
    })