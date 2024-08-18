import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from lambdarank.metrics import max_dcg_k, eval_model
from tqdm import tqdm

def train_model(model,optimizer,train_loader,test_loader,epochs,batch_size,ps_rate,per_query_sample_size,device,writer):
    """
    Trains a ranking model using a Learn to Rank approach, evaluating its performance over a specified number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimization algorithm used to update model parameters.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset, which yields batches of data.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the testing dataset, used for evaluation.
    epochs : int
        Number of full training cycles on the entire dataset.
    batch_size : int
        Number of queries per batch for updating model parameters.
    ps_rate : float
        Probability rate at which the sampled data uses propensity scores rather than actual relevance scores.
    per_query_sample_size : int
        The number of instances to sample per query for training.
    device : torch.device
        The device (CPU or GPU) on which the computations will be performed.
    writer : torch.utils.tensorboard.SummaryWriter
        A writer for logging metrics and training progress to TensorBoard.

    Returns
    -------
    model : torch.nn.Module
        The trained model.

    Notes
    -----
    This function trains a ranking model via grouping query results, applying preference pairs, and optimizing with gradient updates.
    Propensity-based sampling and relevance-based sampling are used interchangeably based on the ps_rate.
    Performance metrics for each epoch are logged using TensorBoard.

    The function is part of a machine learning pipeline for ranking tasks in an e-commerce or similar environment where models
    are trained to automatically rank items based on predicted scores generated from user interaction data.
    """
    alpha = model.alpha
    scheduler = ExponentialLR(optimizer, gamma=1.0)
    for epoch in tqdm(range(1,epochs+1)):
        grad_batch, y_pred_batch = [], []
        query_count = 0
        for query_epoch, train_data in enumerate(train_loader):
            X_train_, y_train_, ps_train_ = train_data
            X_train, y_train, ps_train = X_train_[0], y_train_[0], ps_train_[0]
            if y_train_.sum() == 0.0:
                continue

            y_pred = model(X_train.to(device))
            y_pred_batch.append(y_pred)
            
            with torch.no_grad():
                # if np.random.rand() < ps_rate:
                #     rel_diff = ps_train.view(-1,1) - ps_train.view(-1,1).t()
                #     gain_diff = rel_diff
                # else:
                rel_diff = y_train.view(-1,1) - y_train.view(-1,1).t()
                gain_diff = model.gain[y_train.long()].reshape(-1,1) - model.gain[y_train.long()].reshape(1,-1)
                    
                Sij = (rel_diff > 0).int() - (rel_diff < 0).int()
                Pji = 1.0 / (1.0 + torch.exp(alpha * (y_pred - y_pred.t())))

                delta_ndcg = compute_delta_ndcg(y_pred,y_train,gain_diff,len(y_train),model.gain,device)
                lambda_update = compute_lambda(alpha,delta_ndcg,Sij,Pji,device)
                grad_batch.append(lambda_update)
                
                writer.add_scalar("Lambda/train", lambda_update.detach().mean(), epoch * (query_epoch+1))
                writer.add_scalar("Pji/train", Pji.detach().mean(), epoch * (query_epoch+1))
                writer.add_scalar("Abs_delta_nDCG/train", delta_ndcg.detach().mean(), epoch * (query_epoch+1))
            
            query_count += 1
            if query_count % batch_size == 0:
                for grad,y_pred in zip(grad_batch,y_pred_batch):
                    y_pred.backward(grad)
                optimizer.step()
                model.zero_grad()
                grad_batch, y_pred_batch = [], []
        
        for k in model.eval_at:
            writer.add_scalar(f"nDCG@{k}/train", eval_model(model,train_loader,k,device), epoch)
            writer.add_scalar(f"nDCG@{k}/test", eval_model(model,test_loader,k,device), epoch)
    
        scheduler.step()
        writer.add_scalar(f"LR", scheduler.get_last_lr()[0], epoch)
    return model

def max_ndcg_tensor(y,k,gain=None):
    """
    Computes the maximum possible normalized Discounted Cumulative Gain (nDCG) over the top `k` entries in a tensor `y`.

    Parameters
    ----------
    y : torch.Tensor
        A tensor containing relevance scores from which to compute the nDCG value.
    k : int
        The number of top items to consider for computing the nDCG. If `k` is greater than the number of items in `y`, 
        it will consider the maximum number available.
    gain : torch.Tensor, optional
        A tensor of gains associated with the relevance scores. If provided, these gains will be used instead of the 
        relevance scores in `y` to sort the items for nDCG computation. If `None`, the relevance scores in `y` are used. 
        Default is `None`.

    Returns
    -------
    float
        The maximum nDCG value for the top `k` elements using either the relevance scores or the gain values provided.

    Notes
    -----
    The nDCG is computed as the sum of the relevance scores (or gains) divided by the logarithm of the rank positions 
    (discounting factor), which biases rewards towards the top of the ranked list.
    This function computes the ideal nDCG (i.e., the highest possible nDCG) given the items’ scores. This value can then be used
    to normalize the nDCG computation to obtain scores between 0 and 1.

    Examples
    --------
    >>> y = torch.tensor([2, 0, 3, 0, 4])
    >>> k = 3
    >>> max_ndcg_tensor(y, k)
    tensor(6.8928)

    >>> gains = torch.tensor([10, 20, 30, 40, 50])
    >>> max_ndcg_tensor(y, k, gain=gains)
    tensor(90.2372)
    """
    k = min(len(y),k)
    discount = torch.log2(torch.arange(start=1, end=k+1, step=1) + 1)
    
    if gain is None:
        sorted_label_tensor = y.sort(descending=True)[0][:k]
    else:
        sorted_label_tensor = gain[y.long()].sort(descending=True)[0][:k]
    
    return (sorted_label_tensor / discount).sum()

def compute_delta_ndcg(y_pred,y_train,gain_diff,k,gain,device):
    """
    Computes the gradient of Delta-nDCG based on the predicted and true relevance scores.

    Parameters
    ----------
    y_pred : torch.Tensor
        A tensor containing predicted scores for a batch of items.
    y_train : torch.Tensor
        A tensor containing true relevance scores for a batch of items.
    gain_diff : torch.Tensor
        A pairwise difference matrix of gain values derived from y_train.
    k : int
        The number of top items considered for nDCG computation.
    gain : torch.Tensor
        A tensor of gains associated with the relevance scores. Used for sorting the items 
        for nDCG computation and for calculating the maximum possible nDCG.
    device : torch.device
        The device (e.g., CPU or GPU) to perform computations.

    Returns
    -------
    torch.Tensor
        The absolute Delta-nDCG values, a tensor that represents the change in nDCG due 
        to pairwise swaps of ranks based on the predicted scores.

    Notes
    -----
    Delta-nDCG quantifies the potential change in the nDCG value when the rank position 
    of one item is perturbed. It's calculated for each item in a ranked list and is used to train ranking models.
    
    This function computes Delta-nDCG by taking the difference in discount factors at positions determined 
    by the predicted scores. It normalizes it by the maximum possible nDCG value computed using `max_ndcdg_tensor`.

    Examples
    --------
    >>> y_pred = torch.tensor([0.2, 0.3, 0.1])
    >>> y_train = torch.tensor([2, 0, 1])
    >>> gain_diff = torch.tensor([[0, 2, 1], [-2, 0, -1], [-1, 1, 0]])
    >>> k = 3
    >>> gain = torch.tensor([10, 20, 30])
    >>> device = torch.device('cpu')
    >>> compute_delta_ndcg(y_pred, y_train, gain_diff, k, gain, device)
    tensor([[0.0000, 0.0173, 0.0031],
            [0.0173, 0.0000, 0.0117],
            [0.0031, 0.0117, 0.0000]]])
    """
    rank_order = (y_pred.reshape(-1).argsort(descending=True).argsort() + 1).reshape(-1, 1)
    decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
    ndcg_norm = 1.0 / max_ndcg_tensor(y_train,k,gain)
    delta_ndcg = torch.abs(ndcg_norm * gain_diff.to(device) * decay_diff)
    
    return delta_ndcg

def compute_lambda(alpha,delta_ndcg,Sij,Pji,device):
    """
    Computes the lambda gradient values used for updating the model, based on relevance differences 
    and pairwise probabilities.

    Parameters
    ----------
    alpha : float
        Scaling factor modulating the impact of the difference in the lambda values.
    delta_ndcg : torch.Tensor
        The delta nDCG values for each pair in a batch, representing changes in nDCG due to swapping item ranks.
    Sij : torch.Tensor
        Matrix representing the pairwise preference of items. The element Sij is 1 if item i is more relevant than item j,
        -1 if item i is less relevant, and 0 if they have the same relevance.
    Pji : torch.Tensor
        Matrix of pairwise probabilities that item j is ranked higher than item i.
    device : torch.device
        The device (CPU or GPU) on which computations will be performed.

    Returns
    -------
    torch.Tensor
        The computed lambda values for each item in a batch, which are used to apply gradients for training the model.

    Notes
    -----
    The function computes the lambda values as part of the LambdaRank algorithm, which is based on computing gradients
    of a cost function designed to optimize the nDCG ranking metric. The lambda values are used in a gradient descent
    algorithm to update the weights of the ranking model.
    
    The computation incorporates the following aspects:
    - alpha modulating the relationship between delta_ndcg and the differences in ranking probability and preference,
    - preference differences (Sij),
    - and probability estimates (Pji).

    Examples
    --------
    >>> alpha = 0.001
    >>> delta_ndcg = torch.tensor([0.1, 0.2, 0.15])
    >>> Sij = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    >>> Pji = torch.tensor([[0.5, 0.3, 0.7], [0.7, 0.5, 0.4], [0.3, 0.6, 0.5]])
    >>> device = torch.device('cuda')
    >>> compute_lambda(alpha, delta_ndcg, Sij, Pji, device)
    tensor([[-1.5000e-05],
            [-3.0000e-05],
            [ 5.0000e-05]])
    """
    return (
        (alpha * delta_ndcg) * (0.5 * (1 - Sij.to(device)) - Pji)
    ).sum(dim=1,keepdim=True)