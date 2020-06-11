# Incentives for Federated Learning: a Hypothesis Elicitation Approach

This repository is the official implementation of [Incentives for Federated Learning: a Hypothesis Elicitation Approach] (paper ID: 4657). 


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, seaborn, tqdm, csv, torch


## Training

To train two agents on the MNIST dataset in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python agent_training_mnist.py
```

> 📋For MNIST dataset, agent 1 uses LeNet5 architecture, agent 2 uses CNN13 architecture specified in "basic_model_mnist.py".
To train two agents on the CIFAR-10 dataset in the paper, run this command:

```
CUDA_VISIBLE_DEVICES=0 python agent_training_cifar.py
```

> 📋For CIFAR-10 dataset, agent 1 uses ResNet34 architecture, agent 2 uses CNN13 architecture specified in "resnet_cifar.py" and "basic_model_cifar.py".

## Pre-trained Models

Due to the constrain of file size, we only provide our trained models for MNIST dataset. 

> 📋The trained models for MNIST dataset is in the directory: "trained_models".

## Run Experiments

To reproduce uniform misreport model when there is ground truth for verfication, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_uniform_verification_{dataset}.py
```
To reproduce sparse misreport model when there is ground truth for verfication, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_sparse_verification_{dataset}.py
```
To reproduce uniform misreport model when there is no ground truth for verfication, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_uniform_no_verification_{dataset}.py
```
To reproduce sparse misreport model when there is no ground truth for verfication, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_sparse_no_verification_{dataset}.py
```
To reproduce uniform misreport model with adversarial attacks, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_uniform_attack_{dataset}.py
```
To reproduce sparse misreport model with adversarial attacks, run:

```
CUDA_VISIBLE_DEVICES=0 python runner_sparse_attack_{dataset}.py
```
> 📋More details and hyperparameter settings can be seen in the supplementary materials and the corresponding runners.


## Thanks for watching!
