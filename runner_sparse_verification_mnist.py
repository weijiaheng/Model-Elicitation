import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
import pickle
import dataset
import peer_datasets
import csv
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


class CrossEntropyLossStable(nn.Module):
    '''
    Modified, as the original CrossEntropyLoss are likely be
    blowup when using in the peer term
    '''
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )


def zero_one_peer_one_hot(output, target, peer_output, peer_target, noise_rate, noise_type="Sparse"):
    _, predicted = output.max(1)
    count = 0
    _, store = output.max(1)
    for row in predicted.split(1):
        predicted[count] = add_noise(row, noise_rate, noise_type="Sparse").to(device)
        count += 1
    size_ori = target.size(0)
    correct_ori = predicted.eq(target).sum().item()
    loss_ori = size_ori - correct_ori
    loss_ori /= size_ori
    

    #Peer term
    _, peer_predicted = peer_output.max(1)
    count = 0
    for row in peer_predicted.split(1):
        peer_predicted[count] = add_noise(row, noise_rate, noise_type="Sparse").to(device)
        count += 1
    size_peer = peer_target.size(0)
    correct_peer = peer_predicted.eq(peer_target).sum().item()
    loss_peer = size_peer - correct_peer
    loss_peer /= size_peer
    
    loss = loss_ori - loss_peer
    
    return loss * size_ori


def ce_peer_prob(output, target, peer_output, peer_target, noise_rate, noise_type="Sparse"):
    criterion = CrossEntropyLossStable()
    criterion.cuda(device)
    _, predicted_ori = output.max(1)
    new_prediction = predicted_ori
    count = 0
    _, store_ori = output.max(1)
    for row in predicted_ori.split(1):
        new_prediction[count] = add_noise(row, noise_rate, noise_type="Sparse").to(device)
        if new_prediction[count] != store_ori[count]:
            tmp = output[count][store_ori[count]]
            output[count][store_ori[count]] = output[count][new_prediction[count]]
            output[count][new_prediction[count]] = tmp
        count += 1

    loss_ori = criterion(output, target) / target.size(0)

    _, predicted_peer = peer_output.max(1)
    new_prediction_peer = predicted_peer
    count2 = 0
    _, store_peer = peer_output.max(1)
    for row in predicted_peer.split(1):
        new_prediction_peer[count2] = add_noise(row, noise_rate, noise_type="Sparse").to(device)
        if new_prediction_peer[count2] != store_peer[count2]:
            tmp = peer_output[count2][store_peer[count2]]
            peer_output[count2][store_peer[count2]] = peer_output[count2][new_prediction_peer[count2]]
            peer_output[count2][new_prediction_peer[count2]] = tmp
        count2 += 1
    loss_peer = criterion(peer_output, peer_target) / peer_target.size(0)
    loss = loss_ori - loss_peer
    loss = loss.item()
    return loss * target.size(0)
    

def add_noise(prediction, noise_rate, noise_type="Sparse"):
    r = noise_rate
    conf_matrix = torch.eye(10)
    if noise_type=="Uniform":
        for i in range(10):
            conf_matrix[i][i] = 1 - r * 10
            for j in range(10):
                conf_matrix[i][j] += r
    else:
        conf_matrix[9][1] = r
        conf_matrix[9][9] = 1-r
        conf_matrix[2][0] = r
        conf_matrix[2][2] = 1-r
        conf_matrix[4][7] = r
        conf_matrix[4][4] = 1-r
        conf_matrix[3][5] = r
        conf_matrix[3][3] = 1-r
        conf_matrix[6][8] = r
        conf_matrix[6][6] = 1-r
        conf_matrix[1][9] = r
        conf_matrix[1][1] = 1-r
        conf_matrix[0][2] = r
        conf_matrix[0][0] = 1-r
        conf_matrix[7][4] = r
        conf_matrix[7][7] = 1-r
        conf_matrix[5][3] = r
        conf_matrix[5][5] = 1-r
        conf_matrix[8][6] = r
        conf_matrix[8][8] = 1-r
    prediction = int(np.random.choice(10, 1, p=np.array(conf_matrix[prediction.item()])))
    prediction = torch.tensor(prediction)
    return prediction
        

def calculate_score(noise_rate, noise_type, model, test_dataloader, Loss_zero_one_peer_one_hot, Loss_ce_peer_prob):
    
    for i, data in enumerate(tqdm(test_dataloader), 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        output = output.to(device)
        datasize = labels.size(0)
       
        peer_dataloader1 = peer_datasets.mnist(datasize, datasize, IMG_SIZE)
        for j, data_p1 in enumerate(tqdm(peer_dataloader1), 0):
            inputs_p1, labels_p1 = data_p1
            inputs_p1 = inputs_p1.to(device)
            output_p1 = model(inputs_p1)
            output_p1.to(device)
        
        
        peer_dataloader2 = peer_datasets.mnist(datasize, datasize, IMG_SIZE)
        for k, data_p2 in enumerate(tqdm(peer_dataloader2), 0):
            inputs_p2, labels_p2 = data_p2
            labels_p2 = labels_p2.to(device)
        

        Loss_zero_one_peer_one_hot += zero_one_peer_one_hot(output, labels, output_p1, labels_p2, noise_rate, noise_type="Sparse")
        Loss_ce_peer_prob += ce_peer_prob(output, labels, output_p1, labels_p2, noise_rate, noise_type="Sparse")
    Loss_zero_one_peer_one_hot /= 10000
    Loss_ce_peer_prob /= 10000
    return -Loss_zero_one_peer_one_hot, -Loss_ce_peer_prob
        


    
if __name__ == '__main__':
    IMG_SIZE = 32
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
    batch_size = 64
    test_indices = list(range(10000))
    
    # old_noise_rate_list = misreport rate
    old_noise_rate_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    noise_rate_list = [i for i in old_noise_rate_list]
    noise_type = "Sparse"
    test_dataloader = peer_datasets.load_agent_mnist(batch_size, test_indices, img_size=(32, 32))
    # Choose agent number: 1 or 2
    agent_number = 1
    if agent_number == 1:
        model = Model1().to(device)
        model_path = './trained_models/mnist_f1_exp1_MNIST_0.5_True_dif_model.pth'
    else:
        model = Model2().to(device)
        model_path = './trained_models/mnist_f2_exp1_MNIST_0.5_True_dif_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda(device)
    Loss_zero_one_peer_one_hot = 0.0
    Loss_ce_peer_prob = 0.0
    s1 = []
    s2 = []
    exp_mark = f"MNIST_verification_agent{agent_number}_{noise_type}"
    for i in range(len(noise_rate_list)):
        noise_rate = noise_rate_list[i]
        score_zero_one_peer_one_hot, score_ce_peer_prob= calculate_score(noise_rate, noise_type, model, test_dataloader, Loss_zero_one_peer_one_hot,Loss_ce_peer_prob)
        print(f"Score_zero_one_peer_one_hot{score_zero_one_peer_one_hot}")
        print(f"Score_ce_peer_prob_{score_ce_peer_prob}")
        s1.append(score_zero_one_peer_one_hot)
        s2.append(score_ce_peer_prob)
    score_zero_one_peer_one_hot = s1
    score_ce_peer_prob = s2
    old_noise_rate_list = pd.DataFrame(old_noise_rate_list)
    score_zero_one_peer_one_hot = pd.DataFrame(score_zero_one_peer_one_hot)
    score_ce_peer_prob = pd.DataFrame(score_ce_peer_prob)
    df = pd.concat([old_noise_rate_list, score_zero_one_peer_one_hot, score_ce_peer_prob], axis=1)
    df.columns = ["misreport rate","zero-one", "CE"]
    os.makedirs("./MNIST", exist_ok=True)
    df.to_csv(f"./MNIST/verification_noise_type_{noise_type}_agent{agent_number}.csv", index = False)
