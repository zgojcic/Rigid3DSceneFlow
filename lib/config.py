from lib.model.rigid_3d_sf import MinkowskiFlow
from lib.trainer import MEFlowTrainer
import torch
import yaml
import torch.optim as optim

model_dict = {
     'ME': MinkowskiFlow,
}

trainer_dict = {
     'ME': MEFlowTrainer,
}


def get_model(cfg):
    ''' 
    Gets the model instance based on the input paramters.
    Args:
        cfg (dict): config dictionary
    
    Returns:
        model (nn.Module): torch model initialized with the input params
    '''

    method = cfg['method']['backbone']

    model = model_dict[method](cfg)

    return model


def get_trainer(cfg, model, device):
    ''' 
    Returns a trainer instance.
    Args:
        cfg (dict): config dictionary
        model (nn.Module): the model used for training
        device: torch device

    Returns:
        trainer (trainer instance): trainer instance used to train the network
    '''
    
    method = cfg['method']['backbone']
    trainer = trainer_dict[method](cfg, model, device)


    return trainer


def get_optimizer(cfg, model):
    ''' 
    Returns an optimizer instance.
    Args:
        cfg (dict): config dictionary
        model (nn.Module): the model used for training

    Returns:
        optimizer (optimizer instance): optimizer used to train the network
    '''
    
    method = cfg['optimizer']['alg']

    if method == "SGD":
        optimizer = getattr(optim, method)(model.parameters(), lr=cfg['optimizer']['learning_rate'],
                                                        momentum=cfg['optimizer']['momentum'],
                                                        weight_decay=cfg['optimizer']['weight_decay'])

    elif method == "Adam":
        optimizer = getattr(optim, method)(model.parameters(), lr=cfg['optimizer']['learning_rate'],
                                                        weight_decay=cfg['optimizer']['weight_decay'])
    else: 
        print("{} optimizer is not implemented, must be one of the [SGD, Adam]".format(method))

    return optimizer


def get_scheduler(cfg, optimizer):
    ''' 
    Returns a learning rate scheduler
    Args:
        cfg (dict): config dictionary
        optimizer (torch.optim): optimizer used for training the network

    Returns:
        scheduler (optimizer instance): learning rate scheduler
    '''
    
    method = cfg['optimizer']['scheduler']

    if method == "ExponentialLR":
        scheduler = getattr(optim.lr_scheduler, method)(optimizer, gamma=cfg['optimizer']['exp_gamma'])
    else: 
        print("{} scheduler is not implemented, must be one of the [ExponentialLR]".format(method))

    return scheduler



# General config
def get_config(path, default_path='./configs/default.yaml'):
    ''' 
    Loads config file.
    
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' 
    Update two config dictionaries recursively.
    
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v