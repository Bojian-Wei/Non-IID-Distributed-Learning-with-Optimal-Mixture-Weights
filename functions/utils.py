import os
import numpy as np
from math import ceil
from random import Random
import pickle

import torch
from torch import float32
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as IMG_models
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset
import json
# from models import *

class Logger(object):
    def __init__(self,filename):
        self.log=open(filename,'w')
    def write(self,content):
        self.log.write(content)
        self.log.flush()

def is_regression(dataest):
    if dataest == 'abalone' or dataest == 'abalone' or dataest == 'cadata' or dataest == 'cpusmall' or dataest == 'space_ga' :
        return True

class svmlight_data(Dataset):
    def __init__(self, data_name, root_dir='../FedAMW/datasets/', transform=None, target_transform=None):
        self.inputs, self.outputs = load_svmlight_file(root_dir + data_name)
        if is_regression(data_name):
            self.outputs = 100*(self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
        else:
            if len(set(self.outputs)) == 2:
                self.outputs = (self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
            elif len(set(self.outputs))  > 2:
                self.outputs -= self.outputs.min()
        # self.inputs = self.inputs.toarray()
        # self.outputs = self.outputs.toarray()
        self.transform = transform
        self.target_transform = target_transform
        self.data_name = data_name

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        sample = torch.tensor(self.inputs[idx].A, dtype=torch.float32).view(-1)
        if is_regression(self.data_name):
            label = torch.tensor(self.outputs[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.outputs[idx], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化处理
    x = x.reshape((-1,)) # 拉平
    x = torch.tensor(x)
    return x

def load_synthetic_data(device, synthetic_seting = {'alpha':0, 'beta':1, 'd':10, 'local_size': 500, 'partitions': 20}):
    X_train, y_train, X_test, y_test, data_hete, model_hete = generate_synthetic(synthetic_seting['alpha'], synthetic_seting['beta'], synthetic_seting['d'], synthetic_seting['local_size'], synthetic_seting['partitions'])

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    return X_train, y_train, X_test, y_test, data_hete, model_hete

def load_data(dataset_name, batch_size = 32):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../FedAMW/data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.CIFAR10(root='../FedAMW/data', train=False,
                                              download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [45000, 5000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=5000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 3072, 10
    elif dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../FedAMW/data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.MNIST(root='../FedAMW/data', train=False,
                                               download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [54000, 6000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=6000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 784, 10
    else:
        trainset = svmlight_data(dataset_name)
        trainset, testset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset) - int(len(trainset)*0.8)])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
        feature_size = trainset.dataset.inputs.shape[1]
        class_size = len(set(trainset.dataset.outputs))
        return trainloader, testloader, testloader, feature_size, 1 if is_regression(dataset_name) else class_size
    

def load_full_data(dataset_name, num_partitions = 10, alpha = 0.1):
    partition_sizes = [1.0 / num_partitions for _ in range(num_partitions)]
    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../FedAMW/data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.MNIST(root='../FedAMW/data', train=False,
                                               download=True, transform=data_tf)
        if alpha != -1:
            index_partitions, _ = get_Dirichlet_distribution(trainset.targets.numpy(), partition_sizes, alpha)
        else:
            index_partitions = np.array_split(np.random.permutation(len(trainset.targets)), num_partitions)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        return trainloader, testloader, index_partitions, 784, 10
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../FedAMW/data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.CIFAR10(root='../FedAMW/data', train=False,
                                              download=True, transform=data_tf)
        if alpha != -1:
            index_partitions, _ = get_Dirichlet_distribution(np.array(trainset.targets), partition_sizes, alpha)
        else:
            index_partitions = np.array_split(np.random.permutation(len(trainset.targets)), num_partitions)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        return trainloader, testloader, index_partitions, 3072, 10
    else:
        trainset = svmlight_data(dataset_name)
        testset = svmlight_data(dataset_name + '.t')
        if alpha != -1:
            index_partitions, _ = get_Dirichlet_distribution(trainset.outputs, partition_sizes, alpha)
        else:
            index_partitions = np.array_split(np.random.permutation(len(trainset.outputs)), num_partitions)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

        feature_size = trainset.inputs.shape[1]
        class_size = len(set(trainset.outputs))
        return trainloader, testloader, index_partitions, feature_size, 1 if is_regression(dataset_name) else class_size


def select_model(ds, num_class, args):
    # if args.model == 'VGG':
    #     model = vgg11()
    # elif args.model == 'Linear':
    if True:
        class LM(nn.Module):
            def __init__(self, inputsize, outputsize):
                super(LM,self).__init__()
                self.fc = nn.Linear(inputsize,outputsize,bias=False)
            def forward(self,x):
                return self.fc(x)
        model = LM(ds,num_class)
    return model

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))

def generate_synthetic(alpha, beta, d, local_size, partitions):
    if local_size == 0:
        samples_per_user = np.random.lognormal(4, 2, partitions).astype(int) + 50
    else:
        samples_per_user = np.zeros(partitions).astype(int) + local_size
    print('>>> Sample per user: {}'.format(samples_per_user.tolist()))

    num_train = sum(samples_per_user)
    num_test = int(num_train/4)
    X_train = np.zeros((partitions, local_size, d))
    y_train = np.zeros((partitions, local_size))

    # prior for parameters
    u = np.random.normal(0, alpha, partitions)
    v = np.random.normal(0, beta, partitions)

    # testing data from the global distribution
    X_test = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_test)
    w_target = np.ones(d)
    y_test = np.min([-np.dot(X_test, w_target), -np.dot(X_test, w_target)], axis=0)

    model_hete = 0
    for i in range(partitions):
        xx = np.random.multivariate_normal(np.ones(d)*u[i], np.eye(d), samples_per_user[i])
        ww = np.random.multivariate_normal(np.ones(d), np.eye(d)*v[i])
        yy = np.min([-np.dot(xx, ww), -np.dot(xx, ww)], axis=0) + np.random.normal(0, 0.2, samples_per_user[i])
        yy_target = np.min([-np.dot(xx, w_target), -np.dot(xx, w_target)], axis=0)
        model_hete += np.linalg.norm(yy - yy_target) / num_train

        X_train[i] = xx
        y_train[i] = yy
        # print("{}-th users has {} exampls".format(i, len(y_split[i])))

    
    data_hete = 0
    X_train_global = X_train.reshape(-1, d)
    C_global = np.matmul(X_train_global.T, X_train_global) / X_train_global.shape[0]
    for i in range(partitions):
        C_local = np.matmul(X_train[i].T, X_train[i]) / X_train[i].shape[0]
        data_hete += np.linalg.norm(C_global - C_local) / partitions

    print("Data heterogeneity: {}, model heterogeneity: {}".format(data_hete, model_hete))

    return X_train, y_train, X_test, y_test, data_hete, model_hete

def get_Dirichlet_distribution(labels, psizes = [0.7, 0.2, 0.1], alpha = 0.1):
    n_nets = len(psizes)
    K = len(set(labels))
    labelList = np.array(labels)
    min_size = 0
    N = len(labelList)
    np.random.seed(2020)

    net_dataidx_map = {}
    while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)]) + 1/len(idx_k)
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))

    return idx_batch, net_cls_counts

def check_significance(test_arr, best_arr):
    difference = best_arr - test_arr
    return np.mean(difference) / (np.std(difference) / np.sqrt(len(best_arr))) > 1.812

def print_acc(matrix):
    best_index = np.argmax(np.mean(matrix, axis=1))
    best_row = matrix[best_index, :]
    output_str = ''
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        if i == best_index:
            output_str += '&\\textbf{{{:.2f}$\pm${:.2f}}} '.format(row.mean(), row.std())
        elif check_significance(row, best_row):
            output_str += '&{:.2f}$\pm${:.2f} '.format(row.mean(), row.std())
        else:
            output_str += '&\\underline{{{:.2f}$\pm${:.2f}}} '.format(row.mean(), row.std())
    return output_str

def print_time(matrix):
    best_index = np.argmin(np.mean(matrix, axis=1))
    output_str = ''
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        if i == best_index:
            output_str += '&\\textbf{{{:.2f}}} '.format(row.mean())
        else:
            output_str += '&{:.2f} '.format(row.mean())
    return output_str