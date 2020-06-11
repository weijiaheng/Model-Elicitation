import os
import numpy as np
import pandas as pd
import pickle
import csv
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

# MNIST dataloader
def mnist(batch_size, d_ratio, img_size=32,  dif_data = False):
    # Configure data loader
    os.makedirs("data/mnist", exist_ok=True)
    
    train_dataset = datasets.MNIST(
        "data/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    # Split the data
    datasize = 50000
    indices = list(range(datasize))
    np.random.seed(10086)
    np.random.shuffle(indices)
    split = int(np.floor(d_ratio * datasize))
    train_indices_1 = indices[:split]
    np.random.seed(12315)
    np.random.shuffle(indices)
    train_indices_2 = indices[:split]
    sampler1 = SubsetRandomSampler(train_indices_1)
    sampler2 = SubsetRandomSampler(train_indices_2)
    if not dif_data:
        sampler2 = sampler1
    
    # train_dataloader for agent 1
    train_dataloader1 = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler1)
    
    # train_dataloader for agent 2
    train_dataloader2 = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler2)
        
    test_dataset = datasets.MNIST(
           "data/mnist",
           train=False,
           download=True,
           transform=transforms.Compose(
               [transforms.Resize(img_size), transforms.ToTensor(),  transforms.Normalize([0.5], [0.5])]
           ),
       )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)
    
    df1 = pd.DataFrame(train_indices_1)
    df2 = pd.DataFrame(train_indices_2)
    df = pd.concat([df1, df2], axis=1)
    df.columns = ["f1_selected_index", "f2_selected_index"]
    df.to_csv("saved_idx_MNIST_Exp1.csv", index = False)
    
    return train_dataloader1, train_dataloader2, test_dataloader


# CIFAR-10 dataloader
def cifar10(batch_size, d_ratio, img_size=(32, 32), dif_data = False):
    # Configure data loader
    os.makedirs("data/cifar10", exist_ok=True)

    train_dataset = datasets.CIFAR10(
        "data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    )
    # Split the data
    datasize = 50000
    indices = list(range(datasize))
    np.random.seed(10086)
    np.random.shuffle(indices)
    split = int(np.floor(d_ratio * datasize))
    train_indices_1 = indices[:split]
    np.random.seed(12315)
    np.random.shuffle(indices)
    train_indices_2 = indices[:split]
    sampler1 = SubsetRandomSampler(train_indices_1)
    sampler2 = SubsetRandomSampler(train_indices_2)
    if not dif_data:
        sampler2 = sampler1
    
    # train_dataloader for agent 1
    train_dataloader1 = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler1)
    
    # train_dataloader for agent 2
    train_dataloader2 = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler2)
        
    test_dataset = datasets.CIFAR10(
           "data/cifar10",
           train=False,
           download=True,
           transform=transforms.Compose(
               [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
           ),
       )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    df1 = pd.DataFrame(train_indices_1)
    df2 = pd.DataFrame(train_indices_2)
    df = pd.concat([df1, df2], axis=1)
    df.columns = ["f1_selected_index", "f2_selected_index"]
    df.to_csv("saved_idx_CIFAR_Exp1.csv", index = False)
    return train_dataloader1, train_dataloader2, test_dataloader

