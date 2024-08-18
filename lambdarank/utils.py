import torch

def device_loader():
    """
    Get available number of GPUs and device name.

    Returns
    -------
    tuple of (int, str)
        A tuple where the first element is the number of available GPUs 
        and the second element is the name of the device.
    """
    ngpu = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return ngpu, device

def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)