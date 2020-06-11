import os
import sys
import pandas as pd
import numpy as np
import pickle
import dataset
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from basic_model_mnist import *


CUDA = True if torch.cuda.is_available() else False
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(dataloader), 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        output = model(inputs)
        _, predicted = output.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total

    return accuracy

def train(train_dataloader1, train_dataloader2, test_dataloader, f1, f2, exp_mark):
    """
    :param f1: Agent 1
    :param f2: Agent 2
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()

    if CUDA:
        f1.to(device)
        f2.to(device)
        criterion.to(device)
        
    learning_rate = 0.1
    # ----------
       #  Training
       # ----------
    exp_path = f"./{exp_mark}"
    os.mkdir(exp_path)
    os.makedirs("./trained_models", exist_ok=True)
    PATH1 = f"./trained_models/mnist_f1_exp1_{exp_mark}.pth"
    PATH2 = f"./trained_models/mnist_f2_exp1_{exp_mark}.pth"
    for epoch in range(NUM_EPOCHS):
        # Optimizers
        optimizer_f1 = optim.SGD(f1.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        optimizer_f2 = optim.SGD(f2.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        if epoch >= 20:
            learning_rate = 0.01
        if epoch >= 40:
            learning_rate = 0.001
        if epoch >= 60:
            learning_rate = 0.0001
        if epoch >= 80:
            learning_rate = 0.00001
        train_loss_f1 = []
        train_acc_f1 = []
        test_acc_f1 = []
        train_loss_f2 = []
        train_acc_f2 = []
        test_acc_f2 = []
        
        running_loss_f1 = 0.0
        running_loss_f2 = 0.0
        # train f1
        for i, data in enumerate(tqdm(train_dataloader1), 0):
            inputs_f1, labels_f1 = data
            inputs_f1 = inputs_f1.to(device)
            labels_f1 = labels_f1.to(device)
            inputs_f1, labels_f1 = Variable(inputs_f1), Variable(labels_f1)

            # ---- f1 training ----
            # forward + backward + optimize
            outputs_f1 = f1(inputs_f1)
            loss_f1 = criterion(outputs_f1, labels_f1)
            # zero the parameter gradients
            optimizer_f1.zero_grad()
            loss_f1.backward()
            optimizer_f1.step()
            running_loss_f1 += loss_f1.item()
        running_loss_f1 /= len(train_dataloader1)
        train_loss_f1.append(running_loss_f1)
        acc1 = evaluate(f1, train_dataloader1)
        train_acc_f1.append(acc1)
        
        # train f2
        for i, data in enumerate(tqdm(train_dataloader2), 0):
            inputs_f2, labels_f2 = data
            inputs_f2 = inputs_f2.to(device)
            labels_f2 = labels_f2.to(device)
            inputs_f2, labels_f2 = Variable(inputs_f2), Variable(labels_f2)
            batches_done_f2 = epoch * len(train_dataloader2) + i

            # ---- f1 training ----
            # forward + backward + optimize
            outputs_f2 = f2(inputs_f2)
            loss_f2 = criterion(outputs_f2, labels_f2)
            # zero the parameter gradients
            optimizer_f2.zero_grad()
            loss_f2.backward()
            optimizer_f2.step()
            running_loss_f2 += loss_f2.item()
        running_loss_f2 /= len(train_dataloader2)
        train_loss_f2.append(running_loss_f2)
        acc2 = evaluate(f2, train_dataloader2)
        train_acc_f2.append(acc2)
        t_acc1 = evaluate(f1, test_dataloader)
        t_acc2 = evaluate(f2, test_dataloader)
        test_acc_f1.append(t_acc1)
        test_acc_f2.append(t_acc2)
        print(f"Epoch: {epoch} | Train_Loss1: {running_loss_f1} | Train_Acc1: {acc1} | Test_Acc1: {t_acc1}")
        print(f"Epoch: {epoch} | Train_Loss2: {running_loss_f2} | Train_Acc2: {acc2} | Test_Acc2: {t_acc2}")

    print('==> Finished Training ...')
    
    # Save model
    torch.save(f1.state_dict(), PATH1)
    torch.save(f2.state_dict(), PATH2)
    epoch_num = [i for i in range(NUM_EPOCHS)]
    # Save statistics
    epoch_num = pd.DataFrame(epoch_num)
    train_loss_f1 = pd.DataFrame(train_loss_f1)
    train_acc_f1 = pd.DataFrame(train_acc_f1)
    test_acc_f1 = pd.DataFrame(test_acc_f1)
    df1 = pd.concat([epoch_num, train_loss_f1], axis=1)
    df1 = pd.concat([df1, train_acc_f1], axis=1)
    df1 = pd.concat([df1, test_acc_f1], axis=1)
    df1.columns = ["Epoch", "Train_Loss", "Train_Acc", "Test_Acc"]
    df1.to_csv(f"./{exp_mark}/agent1.csv", index = False)
    
    train_loss_f2 = pd.DataFrame(train_loss_f2)
    train_acc_f2 = pd.DataFrame(train_acc_f2)
    test_acc_f2 = pd.DataFrame(test_acc_f2)
    df2 = pd.concat([epoch_num, train_loss_f2], axis=1)
    df2 = pd.concat([df2, train_acc_f2], axis=1)
    df2 = pd.concat([df2, test_acc_f2], axis=1)
    df2.columns = ["Epoch", "Train_Loss", "Train_Acc", "Test_Acc"]
    df2.to_csv(f"./{exp_mark}/agent2.csv", index = False)
    
    
if __name__ == '__main__':
    NUM_EPOCHS = 5
    BATCH_SIZE = 128
    IMG_SIZE = 32
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
    d_ratio = 0.5    # the sampled percentage of original train dataset for either agent
    dif_data = True  # agent 1 and 2 trained on the same subset or not
    f1 = Model1().to(device)     # Choose the model for agent 1
    f2 = Model2().to(device)     # Choose the model for agent 2
    exp_mark = f"MNIST_{d_ratio}_{dif_data}_dif_model"
    # Get the dataloader
    train_dataloader1, train_dataloader2, test_dataloader = dataset.mnist(BATCH_SIZE, d_ratio, IMG_SIZE, dif_data = False)
    # Start training...
    train(train_dataloader1, train_dataloader2, test_dataloader, f1, f2, exp_mark)

