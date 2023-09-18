import numpy as np

import random

import torch
from torchvision import datasets
from torchvision import transforms

import scipy.io as sio

def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255])
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)
    
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader

def get_test_loader(data_dir,
                    batch_size,
                    num_workers=1,
                    pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_data_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size], torch.Generator().manual_seed(random_seed))
   
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    # the batch_size for testing is set as the valid_size
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_data_loader_from_mat(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True,
                     fold_num='1'):
    
    # Load data
    # mat_content = sio.loadmat(data_dir)
    # Each fold has a .mat file
    mat_content = sio.loadmat(data_dir)

    # Read data
    train_data = mat_content['train_data_Fold_'+fold_num]
    train_label = mat_content['train_label_Fold_'+fold_num]
    test_data = mat_content['test_data_Fold_'+fold_num]
    test_label = mat_content['test_label_Fold_'+fold_num]

    # To Tensor
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label.ravel()).long()
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label.ravel()).long()

    train_set = torch.utils.data.TensorDataset(train_data, train_label)
    test_set = torch.utils.data.TensorDataset(test_data, test_label)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=False)

    return train_loader, test_loader

def get_test_loader_from_mat(data_dir,
                     batch_size,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=False,
                     fold_num='1'):
    
    # Load data
    # mat_content = sio.loadmat(data_dir)
    # Each fold has a .mat file
    mat_content = sio.loadmat(data_dir)

    # Read data
    test_data = mat_content['test_data_Fold_'+fold_num]
    test_label = mat_content['test_label_Fold_'+fold_num]

    # To Tensor
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label.ravel()).long()

    test_set = torch.utils.data.TensorDataset(test_data, test_label)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader

def get_dataset(data_dir):

    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)

    return dataset