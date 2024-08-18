# LambdaRank

**WIP**: the code works, I just need to finish the setup of requirements and stuff.

This repository contains an implementation the Learn-to-Rank algorithm LambdaRANK.

## Usage

1. Set the datasets in `data` folder in the format:
  - Train: `X_train.npy`, `y_train.npy`, `q_train.npy`, `ps_train.npy`;
  - Test: `X_test.npy`, `y_test.npy`, `q_test.npy`, `ps_test.npy`.
2. Set a configuration file in `configs` folder.
3. Run `python app.py --config <config-file-name>`.
4. Models are saved in `models` folder.
5. Tensorboard logs are saved in `runs` folder.

Example:
```bash
$ python app.py --config config_1.json
>>> Device: cuda:0
>>> Number of trainable parameters: 15745
>>> 100%|███████████████████████████████████████████| 25/25 [15:41<00:00, 37.68s/it]
>>> ndcg@1 train/test: 0.7806/0.7411
>>> ndcg@3 train/test: 0.7548/0.7495
>>> ndcg@10 train/test: 0.7582/0.7365
>>> ndcg@50 train/test: 0.7521/0.7449
>>> ndcg@500 train/test: 0.6933/0.6850
>>> ndcg@1000 train/test: 0.7894/0.7721
```

## Dataset

1. `X` and `y`: are the same as always, features matrix and label vector. The labels in `y` must be sequential integers starting in zero following an increasing order of relevance.
2. `q`: is a vector (with the same format as `y`) where each element $q_{i}$ is a string identifier (not necessarily the query text, you can use a hashcode) indicating the query for the observations $X_{i}$ and $y_{i}$. So, the dataset must be created in a way that a label is assigned to each `(query,item)` pair, $(q_{i},X_{i}) \rightarrow y_{i}$.  
    Note that elements in `q` should repeat, since LambdaRank only makes pairwise comparisons of items associated with the same query.
3. `ps`: stands for propensity score (or soft label). It is a vector with the same size as `y` and has a similar meaning too. It is a different kind of label that can be used in training, unlike the labels in `y`, these labels can be floats. If you do not have this vector available, make sure to set the following in your config file:
    - `training_parameters > ps_rate` to `0.0`
    - `data_parameters > train_data > soft_label_file` the same `y_train` file
    - `data_parameters > test_data > soft_label_file` the same `y_test` file

## Configurations

- `run_name` (str): identifier of the run used as folder name in `runs` folder.
- `data_parameters`:
    - `data_path` (str): path indicating the `data` folder.
        - `train_data`:
            - `features_file (str)`: name of the `X_train` file.
            - `label_file (str)`:  name of the `y_train` file.
            - `query_file (str)`:  name of the `q_train` file.
            - `soft_label_file (str)`:  name of the `ps_train` file.
        - `test_data`: same pattern as `train_data`.
        
- `train_parameters`:
    - `training_epochs` (int): number of training epochs.
    - `query_batch_size` (int): number of iterations waited to update model weights.
    - `ps_rate` (float): a number between zero and one indicating the percentage of the iterations that will use the soft label. Set this number to zero for a full conventional LambdaRank.
    - `per_query_sample_size` (int): a number indicating the size of data randomly sampled for each query loop in training. This is specially useful if:
        - the average number of items per query is high (50+);
        - or the average number of relevant items per query is close to half the value of the `k` you want to optimize in your `nDCG@k`, i.e. close to half the size of the list the model will ideally rank.
    Decreasing this value will speedup training and doing so, we recommend increasing the number of epochs to increase model generalization.
    
- `model_configs`:
    - `alpha` (float): is a positive hyperparameter used to compute the lambdas:
    $$\lambda_{ij} = \alpha \left( \frac{1}{2}\left( 1 - S_{ij} \right) - \frac{1}{1 + e^{ \alpha \left( s_{i} - s_{j} \right)}}\right).$$
        Default value is `1.0`. You can decrease it if the number of items per query is very high.
    - `train_label_gain` (list(ints)): is a list containing the gains for each label in order of relevance. The label in `y` vector is used as an index of this list. ex: `label_gain=[0,1,3]` means a label: `0` in `y` will have `gain=0`; `1` will have `gain=1`; and `2` will have `gain=3`. Tipically $gain_{i} = 2^{l_{i}} -1$, where $l_{i}$ is the label of observation $i$, i.e. $l_{i}=y_{i}$.
        **Note**: gains could be floats, for now only integers are implemented.
    - `train_eval_at` (list(ints)): the values of `k` to compute `nDCG@k` used in training for train and test loaders. This is used for log purposes.
    - `layers` (list(dicts)): the description of the Pytorch neural network you want to use in sequential mode. Ex:
    ```python
        {"type": "Linear", "params": { "in_features": null,"out_features": 16}},
        {"type": "Dropout","params": {"p": 0.1}},
        {"type": "LeakyReLU","params": {}},
        {"type": "Linear","params": {"in_features": 16,"out_features": 1}}
     ```
- `optimizer_configs` (dict): the description of the Pytorch optimizer used in training. Only `Adam` and `RMSProp` accepted for now. Ex:
    ```python
    {"type": "Adam",
     "params": {
         "lr": 0.0001,
         "betas": [
             0.9,
             0.999
         ],
         "eps": 1e-08,
         "weight_decay": 1e-06,
         "amsgrad": false
         }
     }
    ```

- `models_path` (str): path indicating the `models` folder.
- `model_name` (str): the name of the model (pickle) to be saved. Ex: `"ranker"`. A suffix will be added with the current date.
- `label_gain` list(ints): same as `train_label_gain`, but used only as a final model evaluation.
- `eval_at` list(ints): same as `train_eval_at`, but used only as a final model evaluation.

## LambdaRank trainer's pseudo code

```plaintext
Procedure train_model:
    Input: model, optimizer, train_loader, test_loader, epochs, batch_size, ps_rate, per_query_sample  _size, device, writer
    
    Set alpha to model.alpha
    For each epoch from 1 to epochs:
        Initialize grad_batch and y_pred_batch as empty lists
        Initialize query_count to 0

        For each query batch in train_loader:
            Extract features (X_train_), labels (y_train_), and propensity scores (ps_train_) from the current query batch
            Continue to the next batch if y_train_ sums to zero (skip batches with no positive instances)

            Sample per-query data using X_train_, y_train_, and ps_train_ with per_query_sample_size

            Predict the scores using the model on the sampled data moved to the device

            Append the predictions to y_pred_batch

            If random value < ps_rate:
                Compute pairwise differences using propensity scores (ps_train)
            Else:
                Compute pairwise differences using true labels (y_train) and gains

            Compute pairwise preference matrix (Sij)
            Compute pairwise probability matrix (Pji) based on predicted scores

            Compute Delta nDCG for the current batch
            Compute gradient updates (lambda_updates) using alpha, Delta nDCG, Sij, and Pji

            Append lambda_updates to grad_batch

            Log scalar values lambda, Pji, and Abs_delta_nDCG using writer

            Increment the query_count
            If query_count is a multiple of batch_size:
                Apply gradients from grad_batch to y_pred_batch
                Perform an optimization step
                Reset the model gradients
                Reset grad_batch and y_pred_batch lists

        Loop over different evaluation points (k) defined in model:
            Calculate and log training nDCG at k using eval_model
            Calculate and log testing nDCG at k using eval_model

    Return the trained model

Auxiliary Procedures:
    max_ndcg_tensor:
        Input: labels, k, gain
        Output: Normalized Discounted Cumulative Gain for top k ranks

    compute_delta_ndcg:
        Input: predicted scores, true labels, gain differences, k, gain, device
        Output: Delta nDCG for re-ranking based on predicted scores

    compute_lambda:
        Input: alpha, delta_ndcg, Sij, Pji, device
        Output: Lambdas for gradient updates based on ranking evaluation

    sample_per_query_data:
        Input: features, labels, propensity scores, sample size
        Output: Sampled subset of data per query
```