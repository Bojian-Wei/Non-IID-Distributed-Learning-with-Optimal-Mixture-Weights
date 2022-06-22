from audioop import bias
import torch
import os
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.uniform import Uniform
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random Features
def RFF(d, sigma, D): # n * d => n * D
    m = Uniform(torch.tensor([0.0], device=device), torch.tensor([2 * torch.pi], device=device))
    W = torch.normal(0, sigma, size=(d, D), device = device)
    b = m.sample((1, D)).view(-1, D).to(device)
    return W, b

# Forward feature mapping and load to device
def feature_mapping(X_train, X_test, k_par = 10, D = 200, type = 'gaussian'):
    X_train_FM = torch.zeros(size=(X_train.shape[0], X_train.shape[1], D), device=device)
    if type == 'gaussian':
        W, b = RFF(X_train[0].shape[1], k_par, D)
        for i in range(len(X_train)):
            X_train_FM[i] = 1 / np.sqrt(D) * torch.cos(torch.matmul(X_train[i], W) + b)
        X_test_FM = 1 / np.sqrt(D) * torch.cos(torch.matmul(X_test, W) + b)
        return X_train_FM, X_test_FM
    else:
        return X_train, X_test

# Network
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.classifier = nn.Linear(input_size, output_size, bias=False)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
    def forward(self,instance):
        return self.classifier(instance)

# Learning rates decay
def update_learning_rate(epoch, target_lr, T):
    """
    1) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: target_lr is the reference learning rate from which to scale down
    """
    if epoch == int(T / 2):
        lr = target_lr/10
        # Logger.info('Updating learning rate to {}'.format(lr))
        return lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
    if epoch == int(T * 0.75):
        lr = target_lr/100
        # Logger.info('Updating learning rate to {}'.format(lr))
        return lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
    else:
        return target_lr

# Loss (useless)
def error_estimate(output, target, type = 'regression'):
    mse_loss = nn.MSELoss(reduction='mean')
    if type == 'binary':
        topK = comp_accuracy(output, target)
        target = torch.nn.functional.one_hot(target, output.shape[-1])
        mse = mse_loss(output, target)
        return mse.item(), 1 - topK[0].item() / 100
    elif type == 'multiclass':
        topK = comp_accuracy(output, target)
        target = torch.nn.functional.one_hot(target, output.shape[-1])
        mse = mse_loss(output, target)
        return mse.item(), 1 - topK[0].item() / 100
    elif type == 'regression':
        return mse_loss(output, target).item(), mse_loss(output, target).item()
    else:
        print("Unsupport task type: {}".format(type))

# Accuracy
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

# Loss and accuracy record
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

# Logging
class Logger(object):
    def __init__(self,filename):
        self.log=open(filename,'w')
    def write(self,content):
        self.log.write(content)
        self.log.flush()

# Train
def train_loop(X_train, y_train, type = 'classification', model = None, lr = 0.01, epoch = 2, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01):
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
    global_model = copy.deepcopy(model)
    if type == 'classification':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr)
    model.train()
    for t in range(epoch):
        train_loss = Meter(ptag='Loss')
        train_acc = Meter(ptag='Prec@1')
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            prox_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                prox_term += (w - w_t).norm(2)
            Wtn = 0.0
            for nm, pm in model.named_parameters():
                if nm == 'classifier.weight':
                    Wtn += torch.norm(pm, 'fro')
            if prox and lambda_reg_if:
                loss = criterion(output, label) + mu * prox_term + lambda_reg * Wtn
            if prox and not lambda_reg_if:
                loss = criterion(output, label) + mu * prox_term
            if not prox and lambda_reg_if:
                loss = criterion(output, label) + lambda_reg * Wtn
            if not prox and not lambda_reg_if:
                loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(comp_accuracy(output, label)[0].item(), data.size(0))

    return model.state_dict(), train_loss.avg, train_acc.avg

# Test
def test_loop(X_test, y_test, type = 'classification', model = None, batch_size = 32):
    test_set = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle = True)
    if type == 'classification':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.MSELoss(reduction='mean')
    model.eval()
    test_loss = Meter(ptag='Loss')
    test_acc = Meter(ptag='Acc')
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss.update(loss.item(), data.size(0))
            test_acc.update(comp_accuracy(output, label)[0].item(), data.size(0))
    print('Test loss: {}, \t Test Acc: {}'.format(test_loss.avg, test_acc.avg))
    return test_loss.avg, test_acc.avg

# Centralized learning
def Centralized(X_train, y_train, X_test, y_test, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 200, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01):
    model = MLP(D, num_classes).to(device)
    X_train_all, y_train_all = X_train, y_train
    X_train, y_train = X_train_all[0], y_train_all[0]
    for i in range(len(X_train_all) - 1):
        X_train = torch.concat((X_train, X_train_all[i+1]), 0)    
        y_train = torch.concat((y_train, y_train_all[i+1]), 0)
    weights, train_loss, _ = train_loop(X_train = X_train, y_train = y_train, type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
    model.load_state_dict(weights)
    # X_test_all, y_test_all = X_test, y_test
    # X_test, y_test = X_test_all[0], y_test_all[0]
    # for i in range(len(X_test_all) - 1):
    #     X_test = torch.concat((X_test, X_test_all[i+1]), 0)    
    #     y_test = torch.concat((y_test, y_test_all[i+1]), 0)
    test_loss, test_acc = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
    return train_loss, test_loss, test_acc

# Distributed learning
def Distributed(X_train, y_train, X_test, y_test, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 200, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / sum(num_samples), dtype=torch.float32, device=device)
    local_weights, local_loss = [], []
    for i in range(num_partitions):
        weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
        local_weights.append(copy.deepcopy(weights))
        local_loss.append(local_train_loss)
    train_loss = torch.sum(p * torch.tensor(local_loss, device=device))
    global_weights = local_weights[0]
    for k in global_weights.keys():
        global_weights[k] *= p[0]
        for j in range(1,len(local_weights)):
            global_weights[k] = global_weights[k] + p[j] * local_weights[j][k]
    model.load_state_dict(global_weights)
    test_loss, test_acc = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
    return train_loss, test_loss, test_acc

# FedAMW_OneShot
def FedAMW_OneShot(X_train, y_train, X_test, y_test, validloader, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 200, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = True, lambda_reg = 0.01, round = 100, lr_p = 5e-5):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / np.sum(num_samples), dtype=torch.float32, device=device)
    p.requires_grad_()
    W = torch.zeros((num_classes, D, num_partitions), device=device)
    local_weights, local_loss = [], []
    for i in range(num_partitions):
        weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
        local_weights.append(copy.deepcopy(weights))
        local_loss.append(local_train_loss)
    train_loss = torch.sum(p * torch.tensor(local_loss, device=device))
    g_weights = local_weights[0]
    for k in g_weights.keys():
        W[:,:,0] = g_weights[k]
        for j in range(1,len(local_weights)):
            W[:,:,j] = local_weights[j][k]
    if type == 'classification':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD([p], lr_p)
    test_loss = torch.zeros(round)
    test_acc = torch.zeros(round)
    for t in range(round):
        valid_loss = Meter(ptag='Loss')
        valid_acc = Meter(ptag='Prec@1')
        for data, label in validloader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = torch.matmul(torch.matmul(W.permute(2,0,1), data.T).permute(2,1,0), p)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            valid_loss.update(loss.item(), data.size(0))
            valid_acc.update(comp_accuracy(output, label)[0].item(), data.size(0))
        # print('Round: {} \t valid loss: {} \t valid acc: {}'.format(t, valid_loss.avg, valid_acc.avg))
        global_weights = local_weights[0]
        for k in global_weights.keys():
            global_weights[k] *= p[0]
            for j in range(1,len(local_weights)):
                global_weights[k] = global_weights[k] + p[j] * local_weights[j][k]
        model.load_state_dict(global_weights)
        tl, ta = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
        test_loss[t], test_acc[t] = tl, ta
    return train_loss, test_loss, test_acc

# FedAvg
def FedAvg(X_train, y_train, X_test, y_test, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 2, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01, round = 100):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / sum(num_samples), dtype=torch.float32, device=device)
    test_loss = torch.zeros(round)
    test_acc = torch.zeros(round)
    train_loss = torch.zeros(round)
    for t in range(round):
        lr = update_learning_rate(t, lr, round)
        local_weights, local_loss = [], []
        for i in range(num_partitions):
            weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
            local_weights.append(copy.deepcopy(weights))
            local_loss.append(local_train_loss)
        train_loss[t] = torch.sum(p * torch.tensor(local_loss, device=device))
        global_weights = local_weights[0]
        for k in global_weights.keys():
            global_weights[k] *= p[0]
            for j in range(1,len(local_weights)):
                global_weights[k] = global_weights[k] + p[j] * local_weights[j][k]
        model.load_state_dict(global_weights)
        tl, ta = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
        test_loss[t], test_acc[t] = tl, ta
    return train_loss, test_loss, test_acc

# FedProx
def FedProx(X_train, y_train, X_test, y_test, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 2, batch_size = 32, prox = True, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01, round = 100):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / sum(num_samples), dtype=torch.float32, device=device)
    test_loss = torch.zeros(round)
    test_acc = torch.zeros(round)
    train_loss = torch.zeros(round)
    for t in range(round):
        lr = update_learning_rate(t, lr, round)
        local_weights, local_loss = [], []
        for i in range(num_partitions):
            weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
            local_weights.append(copy.deepcopy(weights))
            local_loss.append(local_train_loss)
        train_loss[t] = torch.sum(p * torch.tensor(local_loss, device=device))
        global_weights = local_weights[0]
        for k in global_weights.keys():
            global_weights[k] *= p[0]
            for j in range(1,len(local_weights)):
                global_weights[k] = global_weights[k] + p[j] * local_weights[j][k]
        model.load_state_dict(global_weights)
        tl, ta = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
        test_loss[t], test_acc[t] = tl, ta
    return train_loss, test_loss, test_acc

# FedNova
def FedNova(X_train, y_train, X_test, y_test, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 2, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = False, lambda_reg = 0.01, round = 100):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / sum(num_samples), dtype=torch.float32, device=device)
    Tau = torch.tensor(num_samples * epoch / batch_size, device=device)
    Tau_eff = torch.sum(Tau * p)
    test_loss = torch.zeros(round)
    test_acc = torch.zeros(round)
    train_loss = torch.zeros(round)
    for t in range(round):
        lr = update_learning_rate(t, lr, round)
        local_weights, local_loss = [], []
        for i in range(num_partitions):
            weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
            local_weights.append(copy.deepcopy(weights))
            local_loss.append(local_train_loss)
        train_loss[t] = torch.sum(p * torch.tensor(local_loss, device=device))
        global_weights = local_weights[0]
        for k in global_weights.keys():
            global_weights[k] *= (p[0] * Tau_eff / Tau[0])
            for j in range(1,len(local_weights)):
                global_weights[k] = global_weights[k] + p[j] * local_weights[j][k] * Tau_eff / Tau[j]
            # global_weights[k] *= Tau_eff
        model.load_state_dict(global_weights)
        tl, ta = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
        test_loss[t], test_acc[t] = tl, ta
    return train_loss, test_loss, test_acc

# FedAMW
def FedAMW(X_train, y_train, X_test, y_test, validloader, type = 'classification', num_classes = 10, D = 200, lr = 0.01, epoch = 2, batch_size = 32, prox = False, mu = 0.1, lambda_reg_if = True, lambda_reg = 0.01, round = 100, lr_p = 5e-5):
    model = MLP(D, num_classes).to(device)
    num_partitions = len(y_train)
    num_samples = np.array([len(y) for y in y_train])
    p = torch.tensor(num_samples / sum(num_samples), dtype=torch.float32, device=device, requires_grad=True)
    # p.requires_grad
    if type == 'classification':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD([p], lr_p, momentum=0.9)
    test_loss = torch.zeros(round)
    test_acc = torch.zeros(round)
    train_loss = torch.zeros(round)
    for t in range(round):
        lr = update_learning_rate(t, lr, round)
        local_weights, local_loss = [], []
        for i in range(num_partitions):
            weights, local_train_loss, _ = train_loop(X_train[i], y_train[i], type = type, model = model, lr = lr, epoch = epoch, batch_size = batch_size, prox = prox, mu = mu, lambda_reg_if = lambda_reg_if, lambda_reg = lambda_reg)
            local_weights.append(copy.deepcopy(weights))
            local_loss.append(local_train_loss)
        train_loss[t] = torch.sum(p * torch.tensor(local_loss, device=device))
        W = torch.zeros((num_classes, D, num_partitions), device=device, requires_grad=False)
        g_weights = local_weights[0]
        for k in g_weights.keys():
            W[:,:,0] += g_weights[k]
            for j in range(1,len(local_weights)):
                W[:,:,j] += local_weights[j][k]
        for _ in range(round):
            valid_loss = Meter(ptag='Loss')
            valid_acc = Meter(ptag='Prec@1')
            for data, label in validloader:
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = torch.matmul(torch.matmul(W.permute(2,0,1), data.T).permute(2,1,0), p)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                valid_loss.update(loss.item(), data.size(0))
                valid_acc.update(comp_accuracy(output, label)[0].item(), data.size(0))
        # print('Round: {} \t valid loss: {} \t valid acc: {}'.format(t, valid_loss.avg, valid_acc.avg))
        global_weights = local_weights[0]
        for k in global_weights.keys():
            global_weights[k] *= p[0]
            for j in range(1,len(local_weights)):
                global_weights[k] = global_weights[k] + p[j] * local_weights[j][k]
        model.load_state_dict(global_weights)
        tl, ta = test_loop(X_test = X_test, y_test = y_test, type = type, model = model, batch_size = batch_size)
        test_loss[t], test_acc[t] = tl, ta
    return train_loss.detach(), test_loss, test_acc