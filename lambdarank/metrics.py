import torch
import numpy as np

def dcg(r,i):
    """
    Compute the Discounted Cumulative Gain (DCG) at rank k for a ranked list of items.

    Parameters
    ----------
    x : list of int or float
        The list of relevance scores for each item.
    k : int
        The rank at which to compute the DCG.

    Returns
    -------
    int or float
        The DCG value at rank k for the given list of relevance scores.
    """
    gain = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 100, 150, 200] # TO DO: remove this and use the one from configs
    return (gain[int(r)])/(np.log2(1+i))

def dcg_k(x,k):
    """
    Compute the Maximum Discounted Cumulative Gain (DCG) at rank k for a ranked list of items.

    Parameters
    ----------
    x : list of int or float
        The list of relevance scores for each item.
    k : int
        The rank at which to compute the maximum DCG.

    Returns
    -------
    int or float
        The maximum DCG value at rank k for the given list of relevance scores.
    """
    result = 0
    for i in range(1,min(k+1,len(x))):
        result += dcg(x[i-1],i)
    return result

def max_dcg_k(x,k):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k for a ranked list of items.

    Parameters
    ----------
    x : list of int or float
        The list of relevance scores for each item.
    k : int
        The rank at which to compute the NDCG.

    Returns
    -------
    float
        The NDCG value at rank k for the given list of relevance scores. Returns 0.0 if the
        ideal DCG is 0 to prevent division by zero.
    """
    x = sorted(x)[::-1]
    return dcg_k(x,k)

def ndcg_k(x,k):
    """
    Add two numbers.

    Parameters
    ----------
    a : int or float
        The first number.
    b : int or float
        The second number.

    Returns
    -------
    int or float
        The sum of the two numbers.
    """
    idcg_k = max_dcg_k(x,k)
    if idcg_k == 0.0:
        return 0.0
    else:
        return dcg_k(x,k)/idcg_k
    
def compute_ndcg(preds,y,k):
    """
    Add two numbers.

    Parameters
    ----------
    a : int or float
        The first number.
    b : int or float
        The second number.

    Returns
    -------
    int or float
        The sum of the two numbers.
    """
    ranked_list = y[preds.ravel().sort(descending=True)[1]].cpu().numpy()
    ndcg_at_k_value = ndcg_k(ranked_list,min(k,len(ranked_list)))
    return ndcg_at_k_value

def eval_model(model,loader,k,device):
    """
    Evaluate a model's performance by computing the average NDCG at a specific rank k over a data loader.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate.
    loader : torch.utils.data.DataLoader
        The DataLoader containing the evaluation data.
    k : int
        The rank at which to compute the NDCG.
    device : torch.device
        The device (CPU or GPU) on which to perform the evaluation.

    Returns
    -------
    float
        The average NDCG value at rank k over the evaluation data.
    """
    loss_ndcg = []
    with torch.no_grad():
        for _, data in enumerate(loader):
            X, y,_ = data
            X, y = X[0], y[0]
            preds = model(X.to(device)).detach().cpu()
            loss_ndcg.append([compute_ndcg(preds,y,k)])
    return np.mean(loss_ndcg)