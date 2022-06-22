"""
Tune Hyperparamters for KRR via NNI.
"""
from functions.tools import feature_mapping, Centralized, Distributed, FedAMW_OneShot, FedAvg, FedProx, FedNova, FedAMW
from functions.utils import load_synthetic_data, load_full_data
from functions.optimal_parameters import get_parameter
import os
import pickle
import argparse
import logging
import nni
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

CUDA_LAUNCH_BLOCKING=1
SAVE = False
logger = logging.getLogger('Tune Hyperparamters')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    
    ### datasets properties
    dataset = args['dataset'] 
    parameter_dic = get_parameter(dataset)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    kernel_type = parameter_dic['kernel_type']

    D = args['D']
    Round = args['round']
    data_dir = args['data_dir']
    local_epoch = int(args['local_epoch'])
    alpha = args['alpha']
    beta = args['beta']
    num_partitions = 50
    batch_size = 32
    alpha_Dirk = 0.01

    ### hyperparameters to tune
    '''同时调核参数k_par和学习率lr'''
    k_par= parameter_dic['kernel_par'] # parameter_dic['kernel_par']
    learning_rate = parameter_dic['lr'] # parameter_dic['lr']
    mu = args['lambda_prox'] # parameter_dic['mu']
    learning_rate_p_os = args['lr_p_os'] # parameter_dic['lr_p_os']
    lambda_reg_os = args['lambda_reg_os'] # parameter_dic['lambda_reg_os']
    learning_rate_p = args['lr_p'] # parameter_dic['lr_p']
    lambda_reg = args['lambda_reg'] # parameter_dic['lambda_reg']

    if dataset == 'synthetic_nonlinear': # Tune parameters for synthetic data
        synthetic_dict = {'alpha':0, 'beta':0, 'd':10, 'local_size': 10000, 'partitions': 1}
        X_train, y_train, X_test, y_test, _, _ = load_synthetic_data(device, synthetic_dict)
        y_train = y_train.reshape(1, y_train.shape[1], num_classes)
        y_test = y_test.reshape(y_test.shape[0], 1)
        X_train_FM, X_test_FM = feature_mapping(X_train, X_test, k_par, D, kernel_type)
        X_train_FM = torch.tensor_split(X_train_FM.reshape(-1, D), num_partitions)
        y_train = torch.tensor_split(y_train.reshape(-1, num_classes), num_partitions)
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
    else: # Tune parameters for real-world dataset
        trainloader, testloader, index_partitions, d, num_classes = load_full_data(dataset, num_partitions, alpha_Dirk)
        X_train, y_train_all = iter(trainloader).next()    
        X_test, y_test = iter(testloader).next()        
        X_train_FM_all, X_test_FM = feature_mapping(X_train.reshape(1, X_train.shape[0], X_train.shape[1]).to(device), X_test.to(device), k_par, D, kernel_type)
        X_train_FM_all = X_train_FM_all.reshape(-1, D)
        X_train_FM, y_train = [], []
        for idx in index_partitions:
            X_train_FM.append(X_train_FM_all[idx,:])
            # y_train.append(torch.nn.functional.one_hot(y_train_all[idx], num_classes).to(device))
            y_train.append(y_train_all[idx].to(device))
            C_j = torch.matmul(X_train_FM[-1].T, X_train_FM[-1]) / len(X_train_FM[-1])
        y_test = y_test.to(device)

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


    # _, Loss, Acc = Centralized(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, False, 0)
    # logger.info("CL --- sigma: {}, Error: {:.5f}, Acc: {:.5f}".format(k_par, Loss, Acc))

    # _, Loss, Acc = Distributed(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, False, 0)
    # logger.info("DL --- sigma: {}, Error: {:.5f}, Acc: {:.5f}".format(k_par, Loss, Acc))

    # _, Loss, Acc = FedAMW_OneShot(X_train_FM, y_train, X_test_FM, y_test, validateloarder, task_type, num_classes, D, learning_rate, local_epoch*Round, batch_size, False, 0, True, lambda_reg_os, Round, learning_rate_p_os)
    # Acc = Acc[-1].item()
    # Loss = Loss[-1].item()
    # logger.info("FedAMW_OneShot --- sigma: {}, Error: {:.5f} Acc: {:.5f}".format(k_par, Loss, Acc))

    # _, Loss, Acc = FedAvg(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, False, 0, Round)
    # Acc = Acc[-1].item()
    # Loss = Loss[-1].item()
    # logger.info("FedAvg --- sigma: {}, Error: {:.5f} Acc: {:.5f}".format(k_par, Loss, Acc))

    # _, Loss, Acc = FedProx(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, True, mu, False, 0, Round)
    # Acc = Acc[-1].item()
    # Loss = Loss[-1].item()
    # logger.info("FedProx --- sigma: {}, Error: {:.5f} Acc: {:.5f}".format(k_par, Loss, Acc))

    # _, Loss, Acc = FedNova(X_train_FM, y_train, X_test_FM, y_test, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, False, 0, Round)
    # Acc = Acc[-1].item()
    # Loss = Loss[-1].item()
    # logger.info("FedNova --- sigma: {}, Error: {:.5f} Acc: {:.5f}".format(k_par, Loss, Acc))

    _, Loss, Acc = FedAMW(X_train_FM, y_train, X_test_FM, y_test, validateloarder, task_type, num_classes, D, learning_rate, local_epoch, batch_size, False, 0, True, lambda_reg, Round, learning_rate_p)
    Acc = Acc[-1].item()
    Loss = Loss[-1].item()
    logger.info("FedAMW --- sigma: {}, Error: {:.5f} Acc: {:.5f}".format(k_par, Loss, Acc))

    nni.report_final_result(Acc)
    logger.debug('Final result is %.5f', Acc)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Tuner')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--dataset", type=str,
                        default='usps', help="dataset name")
    parser.add_argument("--alpha", type=float, 
                        default=0.0, help="data hetegeneity parameter")
    parser.add_argument("--beta", type=float, 
                        default=0.0, help="model hetegeneity")
    parser.add_argument("--D", type=int, default=2000, metavar='N', help='hidden layer size (default: 2000)')
    parser.add_argument("--kernel_par", type=float, default=0.1, help="kernel hyperparameter")
    parser.add_argument('--lambda_reg_os', type=float, default=0.000001, help='regularizer parameter (default: 0.01)')
    parser.add_argument('--lambda_reg', type=float, default=0.000001, help='regularizer parameter (default: 0.01)')
    parser.add_argument('--lambda_prox', type=float, default=0.01, help='regularizer parameter (default: 0.01)')
    parser.add_argument("--data_dir", type=str,
                        default='../FedAMW/datasets', help="data directory")
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lr_p', type=float, default=0.1, metavar='LR_p', help='learning rate for p (default: 0.001)')
    parser.add_argument('--lr_p_os', type=float, default=0.1, metavar='LR_p', help='learning rate for p (default: 0.001)')
    parser.add_argument('--local_epoch', type=int, default=2, help='local update (default: 2)')
    parser.add_argument('--round', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    args, _ = parser.parse_known_args()
    return args

## set parameters in config_gpu.yml 
## and run: nnictl create --config /home/superlj666/Experiment/MM/config.yml --port xxxx
## to shutdown a nni program: nnictl stop xx, where xx is the id
if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)

    except Exception as exception:
        logger.exception(exception)
        raise