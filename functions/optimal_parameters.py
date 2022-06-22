def get_parameter(dataset):
    if dataset == 'mnist':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 60000,
            'dimensional' : 784,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,# opt
            'lambda_reg_os' : 0.000005,# opt
            'lambda_reg' : 0.000005,# opt
            'lambda_prox' : 0.000001,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.001,# opt
            'lr_p' : 0.001# opt
        }
    elif dataset == 'synthetic_nonlinear':
        parameter_dic = {
            'task_type' : 'regression',
            'num_examples' : 10000,
            'dimensional' : 10,
            'num_classes' : 1,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,
            'lambda_reg' : 0.000001, 
            'lambda_prox' : 7e-7,#1e-8, 
            'alpha_Dirk' : 1, 
            'lr' : 0.001,
        }
    elif dataset == 'dna':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 2000,
            'dimensional' : 180,
            'num_classes' : 3,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,# opt
            'lambda_reg_os' : 1e-6,# opt
            'lambda_reg' : 0.01,# opt
            'lambda_prox' : 0.01,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.1,# opt
            'lr_p' : 0.001# opt
        }
    elif dataset == 'letter':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 15000,
            'dimensional' : 16,
            'num_classes' : 26,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,# opt
            'lambda_reg_os' : 0.00005,# opt
            'lambda_reg' : 0.005,# opt
            'lambda_prox' : 0.00005,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.001,# opt
            'lr_p' : 0.0001# opt
        }
    elif dataset == 'pendigits':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 7494,
            'dimensional' : 16,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.01,# opt
            'lambda_reg_os' : 0.005,# 
            'lambda_reg' : 0.01,# opt
            'lambda_prox' : 0.001,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.5,# 
            'lr_p' : 0.0005# opt
        }
    # elif dataset == 'poker':
    #     parameter_dic = {
    #         'task_type' : 'classification',
    #         'num_examples' : 25010,
    #         'dimensional' : 10,
    #         'num_classes' : 10,
    #         'kernel_type' : 'gaussian',
    #         'kernel_par' : 0.01,# 
    #         'lambda_reg_os' : 0.00005,# 
    #         'lambda_reg' : 0.005,# 
    #         'lambda_prox' : 0.00005,# 
    #         'alpha_Dirk' : 0.01,
    #         'lr' : 0.5,# 
    #         'lr_p_os' : 0.001,# 
    #         'lr_p' : 0.0001# 
    #     }
    elif dataset == 'satimage':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 4435,
            'dimensional' : 36,
            'num_classes' : 6,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,# opt
            'lambda_reg_os' : 0.001,# opt
            'lambda_reg' : 0.001,# opt
            'lambda_prox' : 0.0005,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.1,# opt
            'lr_p' : 0.00001# opt
        }
    # elif dataset == 'Sensorless':
    #     parameter_dic = {
    #         'task_type' : 'classification',
    #         'num_examples' : 58509,
    #         'dimensional' : 48,
    #         'num_classes' : 11,
    #         'kernel_type' : 'gaussian',
    #         'kernel_par' : 10,
    #         'lambda_reg' : 0.000001, 
    #         'lambda_prox' : 1e-7,#1e-8, 
    #         'alpha_Dirk' : 1, 
    #         'lr' : 0.001,
    #     }
    # elif dataset == 'shuttle':
    #     parameter_dic = {
    #         'task_type' : 'classification',
    #         'num_examples' : 43500,
    #         'num_classes' : 11,
    #         'dimensional' : 48,
    #         'kernel_type' : 'gaussian',
    #         'kernel_par' : 10,
    #         'lambda_reg' : 0.001, 
    #         'lambda_prox' : 1e-3,#1e-8, 
    #         'alpha_Dirk' : 0.5, 
    #         'lr' : 0.001,
    #     }
    elif dataset == 'usps':
        parameter_dic = {
            'task_type' : 'classification',
            'num_examples' : 7291,
            'dimensional' : 256,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1,# opt
            'lambda_reg_os' : 0.0005,# 
            'lambda_reg' : 0.00005,# opt
            'lambda_prox' : 0.0001,# opt
            'alpha_Dirk' : 0.01,
            'lr' : 0.5,# opt
            'lr_p_os' : 0.005,# opt
            'lr_p' : 0.0005# opt
        }
    else:
        parameter_dic = {
            'task_type' : 'classification',
            'num_classes' : 10,
            'dimensional' : 784,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1, 
            'lambda_reg' : 0.00001, 
            'lambda_prox' : 7e-7, 
            'lr' : 0.001,
        }
    parameter_dic['local_update'] = 100
    return parameter_dic
