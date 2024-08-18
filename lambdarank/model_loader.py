import torch
import torch.nn as nn
import torch.optim as optim

def init_weights(m):
    """
    Initialize the weights of a Linear layer using Xavier normal initialization and set bias to a constant value.

    Parameters
    ----------
    m : torch.nn.Module
        The module for which weights are to be initialized.

    Notes
    -----
    This function is intended to be used as a model apply function to initialize the weights of Linear layers.

    References
    ----------
    - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
      Proceedings of AISTATS, 307-315.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)

def build_model_from_config(model_config):
    """
    Constructs a PyTorch neural network model based on the provided configuration dictionary.

    Parameters
    ----------
    model_config : dict
        A dictionary containing information about the layers to be included in the model. 
        It should have a key "layers" whose value is a list of dictionaries, each representing a layer. 
        Each layer dictionary should contain a "type" key specifying the type of layer (e.g., "Linear", "Conv2d") 
        and optionally a "params" key specifying parameters for that layer (as a dictionary).

    Returns
    -------
    torch.nn.Sequential
        A PyTorch Sequential model containing all the layers specified in the model_config.
    """
    
    layers = []
    for layer_info in model_config["layers"]:
        layer_type = layer_info["type"]
        layer_params = layer_info.get("params", {})
        layer_class = getattr(nn, layer_type)
        layer = layer_class(**layer_params)
        layers.append(layer)
    
    model = nn.Sequential(*layers)
    
    return model

def load_model(input_dim,model_configs,device,ngpu):
    """
    Load a PyTorch model based on input dimensions, model parameters, device, and number of GPUs.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    model_configs : dict
        Dictionary containing configuration for building the model.
    device : torch.device
        The device (CPU or GPU) on which to load the model.
    ngpu : int
        Number of available GPUs.

    Returns
    -------
    torch.nn.Module
        The loaded PyTorch model after applying model configuration and weight initialization.
    """
    
    model_configs['layers'][0]['params']['in_features'] = input_dim
    model = build_model_from_config(model_configs).to(device)
    model.gain = torch.tensor(model_configs['train_label_gain'])
    model.alpha = model_configs['alpha']
    model.eval_at = model_configs['train_eval_at']
    
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))

    model.apply(init_weights)
    
    return model

def load_optim(model,optimizer_configs):
    """
    Load and return the optimizer for the model based on the specified configuration.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the optimizer is loaded.
    optimizer_configs : dict
        The configuration dictionary containing information about the optimizer type and parameters.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer based on the specified configuration. Default to Adam if the optimizer type is not defined.
    """
    optimizer_type = optimizer_configs.get('type')
    
    if optimizer_type == 'Adam':
        optimizer = load_Adam_optim(model,optimizer_configs)
    elif optimizer_type == 'RMSprop':
        optimizer = load_RMSprop_optim(model,optimizer_configs)
    else:
        print("Your optimizer is not defined in `lambdarank/model_definition.py` default Adam will be used instead.")
        optimizer = optim.Adam(params=model.parameters())
    return optimizer
        

def load_Adam_optim(model,optimizer_configs):
    """
    Load and return an Adam optimizer for the model using the specified configuration parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the Adam optimizer is loaded.
    optimizer_configs : dict
        The configuration dictionary containing Adam optimizer parameters.

    Returns
    -------
    torch.optim.Adam
        The Adam optimizer with the specified parameters.
    """
    optimizer = optim.Adam(
        params = model.parameters(),
        **optimizer_configs["params"]
    )
    return optimizer

def load_RMSprop_optim(model,optimizer_configs):
    """
    Load and return an RMSprop optimizer for the model using the specified configuration parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the RMSprop optimizer is loaded.
    optimizer_configs : dict
        The configuration dictionary containing RMSprop optimizer parameters.

    Returns
    -------
    torch.optim.RMSprop
        The RMSprop optimizer with the specified parameters.
    """
    optimizer = optim.RMSprop(
        params = model.parameters(),
        **optimizer_configs["params"]
    )
    return optimizer

def load_model_and_optim(input_dim,model_configs,optimizer_configs,device,ngpu):
    """
    Load a PyTorch model and corresponding optimizer based on input dimensions, model parameters,
    device, and number of GPUs.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    model_configs : dict
        Dictionary containing configuration for building the model.
    optimizer_configs : dict
        Dictionary containing configuration for building the optimizer.
    device : torch.device
        The device (CPU or GPU) on which to load the model.
    ngpu : int
        Number of available GPUs.

    Returns
    -------
    tuple of (torch.nn.Module, torch.optim.Optimizer)
        A tuple containing the loaded PyTorch model and the corresponding optimizer.
    """
    model = load_model(input_dim,model_configs,device,ngpu)
    optimizer = load_optim(model,optimizer_configs)
    
    return model, optimizer