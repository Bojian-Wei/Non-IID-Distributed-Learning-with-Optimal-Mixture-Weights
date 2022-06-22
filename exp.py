"""
Tune Hyperparamters for KRR via NNI.
"""
from functions.tools import feature_mapping, Centralized, Distributed, FedAMW_OneShot, FedAvg, FedProx, FedNova, FedAMW
from functions.utils import load_full_data
from functions.optimal_parameters import get_parameter
import os
import pickle
import argparse
import logging
import nni
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    SAVE = False
    logger = logging.getLogger('Tune Hyperparamters')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(100)
    np.random.seed(100)

    D = 2000                     # dimension of feature mapping
    num_partitions = 50          # number of clients
    data_dir = '../FedAMW/datasets'
    result_dir = './results'
    local_epoch = 2              # local epochs
    Round = 100                  # communication rounds
    batch_size = 32
    n_repeats = 1
    alpha_Dirk = 0.01

    dataset = 'satimage'
    parameter_dic = get_parameter(dataset)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    kernel_type = parameter_dic['kernel_type']
    k_par= parameter_dic['kernel_par']
    learning_rate = parameter_dic['lr']
    learning_rate_p = parameter_dic['lr_p']
    learning_rate_p_os = parameter_dic['lr_p_os']
    mu = parameter_dic['lambda_prox']
    lambda_reg = parameter_dic['lambda_reg']
    lambda_reg_os = parameter_dic['lambda_reg_os']

    train_mat = np.empty((6, Round, n_repeats))
    error_mat = np.empty((6, Round, n_repeats))
    acc_mat = np.empty((6, Round, n_repeats))
    heterogeneity_mat = np.empty(n_repeats)
    for t in range(n_repeats):
        trainloader, testloader, index_partitions, d, num_classes = load_full_data(dataset, num_partitions, alpha_Dirk)
        X_train, y_train_all = iter(trainloader).next()    
        X_test, y_test = iter(testloader).next()
        X_train_FM_all, X_test_FM = feature_mapping(X_train.reshape(1, X_train.shape[0], X_train.shape[1]).to(device), X_test.to(device), k_par, D, kernel_type)

        X_train_FM_all = X_train_FM_all.reshape(-1, D)
        data_hete = 0
        C = torch.matmul(X_train_FM_all.T, X_train_FM_all) / len(X_train_FM_all)
        X_train_FM, y_train = [], []
        for idx in index_partitions:
            X_train_FM.append(X_train_FM_all[idx,:])
            # y_train.append(torch.nn.functional.one_hot(y_train_all[idx], num_classes).to(device))
            y_train.append(y_train_all[idx].to(device))
            C_j = torch.matmul(X_train_FM[-1].T, X_train_FM[-1]) / len(X_train_FM[-1])
            data_hete += len(X_train_FM[-1])/len(X_train_FM_all) * torch.norm(C - C_j)
        y_test = y_test.to(device)
        heterogeneity_mat[t] = data_hete.item()

        X_val_all, y_val_all = [], []
        X_train_all, y_train_all = [], []
        for i in range(num_partitions):
            random_idx = np.arange(X_train_FM[i].shape[0])
            np.random.shuffle(random_idx)
            threshold_idx = int(X_train_FM[i].shape[0] * 0.2)
            val_idx = random_idx[:threshold_idx]
            train_idx = random_idx[threshold_idx:]

            X_val_all.append(X_train_FM[i][val_idx])
            y_val_all.append(y_train[i][val_idx])
            X_train_all.append(X_train_FM[i][train_idx])
            y_train_all.append(y_train[i][train_idx])

        X_val, y_val = X_val_all[0], y_val_all[0]
        X_train_FM = X_train_all
        y_train = y_train_all
        for i in range(len(X_val_all) - 1):
            X_val = torch.concat((X_val, X_val_all[i+1]), 0)
            y_val = torch.concat((y_val, y_val_all[i+1]), 0)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        validateloarder = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True)

        
        cl_loss_train, cl_loss, cl_acc = Centralized(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, False, 0)
        dl_loss_train, dl_loss, dl_acc = Distributed(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, False, 0)
        for i in range(Round):
            train_mat[0, i, t] = cl_loss_train
            error_mat[0, i, t] = cl_loss
            acc_mat[0, i, t] = cl_acc
            train_mat[1, i, t] = dl_loss_train
            error_mat[1, i, t] = dl_loss
            acc_mat[1, i, t] = dl_acc
        amwos_loss_train, amwos_loss, amwos_acc = FedAMW_OneShot(X_train_FM, y_train, X_test_FM, y_test, validateloarder, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, True, lambda_reg_os, Round, learning_rate_p_os)
        for i in range(Round):
            train_mat[2, i, t] = amwos_loss_train
        error_mat[2, :, t] = amwos_loss
        acc_mat[2, :, t] = amwos_acc
        avg_loss_train, avg_loss, avg_acc = FedAvg(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, False, 0, Round)
        train_mat[3, :, t] = avg_loss_train
        error_mat[3, :, t] = avg_loss
        acc_mat[3, :, t] = avg_acc
        prox_loss_train, prox_loss, prox_acc = FedProx(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, True, mu, False, 0, Round)
        train_mat[4, :, t] = prox_loss_train
        error_mat[4, :, t] = prox_loss
        acc_mat[4, :, t] = prox_acc
        # nova_loss, nova_acc = FedNova(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, False, 0, Round)
        # error_mat[5, :, t] = nova_loss
        # acc_mat[5, :, t] = nova_acc
        amw_loss_train, amw_loss, amw_acc = FedAMW(X_train_FM, y_train, X_test_FM, y_test, validateloarder, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, True, lambda_reg, Round, learning_rate_p)
        train_mat[5, :, t] = amw_loss_train
        error_mat[5, :, t] = amw_loss
        acc_mat[5, :, t] = amw_acc
    
    data_ = {
        'epochs': Round,
        'train_loss': train_mat,
        'test_loss': error_mat,
        'test_acc':acc_mat,
        'heterogeneity': heterogeneity_mat,
        'name': ['CL', 'DL', 'FedAMW_OneShot', 'FedAvg', 'FedProx', 'FedAMW']
    }

    result_path = '{}/exp1_{}.pkl'.format(result_dir, dataset)
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)
