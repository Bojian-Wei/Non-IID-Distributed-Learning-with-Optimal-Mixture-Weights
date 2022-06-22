# Non-IID-Distributed-Learning-with-Optimal-Mixture-Weights
Codes of Non-IID Distributed Learning with Optimal Mixture Weights

## This is the official version of **Non-IID Distributed Learning with Optimal Mixture Weights** (ECML-PKDD 2022)

### 1. Requirements
```
Pytorch = 1.7.0
numpy = 1.19.2
scipy = 1.5.2
nni
```

### 2. Structure of the project
```
project
│   README.md   
|   exp.py
|   tune.py
|   config.yml 
│
|───functions
│       │   optimal_parameters.py
|       |   tools.py
|       |   utils.py
```

### 3. Datasets (all the datasets used in this paper can be downloaded by clicking the following link)
- [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)


### 4. Other algorithms
- We define different federated algorithms in tools.py with python-functions. (e.g. the function named 'FedProx' in functions/tools.py)


### 5. Start training
- Tuning hyperparameters: First, you can set a series of intervals for different hyperparameters in config.yml, and choose a specific method for tuning in tune.py. Then, you can conduct the tuning process by the command of nnictl. After you get the optimal hyperparameters, you should copy them to the related keys in optimal_parameters.py.
- Training and Testing: Starting training and testing by running exp.py.
Some details are listed in the .py files.
