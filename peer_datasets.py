import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler



def load_agent_mnist(batch_size, train_indices, img_size=32):
    # Configure data loader
    os.makedirs("data/mnist_peer", exist_ok=True)

    train_dataset = datasets.MNIST(
        "data/mnist_peer",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    
    sampler = SubsetRandomSampler(train_indices)
    agent_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler)
    
    return agent_dataloader



def mnist(batch_size, datasize, img_size=32):
    # Configure data loader
    os.makedirs("data/mnist_peer", exist_ok=True)

    train_dataset = datasets.MNIST(
        "data/mnist_peer",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    # Split the data
    whole_indices = list(range(10000))
    np.random.seed(10086)
    np.random.shuffle(whole_indices)
    split = datasize
    train_indices = whole_indices[:split]
    sampler = SubsetRandomSampler(train_indices)
   
    peer_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler)
    
    return peer_dataloader




def load_agent_cifar10(batch_size, train_indices,  img_size=(32, 32)):
    # Configure data loader
    os.makedirs("data/cifar10_peer", exist_ok=True)

    train_dataset = datasets.CIFAR10(
        "data/cifar10_peer",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    )

    # Split the data
    
    sampler = SubsetRandomSampler(train_indices)
    agent_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler)
     
    return agent_dataloader
        




def cifar10(batch_size, datasize, img_size=(32, 32)):
    # Configure data loader
    os.makedirs("data/cifar10_peer", exist_ok=True)

    train_dataset = datasets.CIFAR10(
        "data/cifar10_peer",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    )

    # Split the data
    whole_indices = list(range(10000))
    np.random.seed(10086)
    np.random.shuffle(whole_indices)
    split = datasize
    train_indices = whole_indices[:split]
    sampler = SubsetRandomSampler(train_indices)
    
    peer_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler)
     
    return peer_dataloader
        

