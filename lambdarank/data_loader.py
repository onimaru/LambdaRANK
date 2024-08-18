import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self,x,y,ps):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.ps = torch.tensor(ps,dtype=torch.float32)
        self.length = self.x.shape[0]
        self.width = self.x.shape[1]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx],self.ps[idx]
    
    def __len__(self):
        return self.length
    
    def __width__(self):
        return self.width
    
def create_query_sampler(q):
    q_sampler = []
    for item in set(q):
        q_sampler.append(
            np.argwhere(
                q==item
            ).ravel().tolist()
        )
    return q_sampler    

def dataloader(X,y,q,ps):
    dataset = CustomDataset(X,y,ps)
    q_sampler = create_query_sampler(q)
    loader = DataLoader(
        dataset, 
        sampler=q_sampler, 
        num_workers=8,
        pin_memory=True
    )
    
    return loader

def _load_numpy_data(data_path,data_params):
    X = np.load(os.path.join(data_path,data_params["features_file"]))
    y = np.load(os.path.join(data_path,data_params["label_file"]))
    ps = np.load(os.path.join(data_path,data_params["soft_label_file"]))
    q = np.load(os.path.join(data_path,data_params["query_file"]),allow_pickle=True)
    
    return X, y, ps, q

def load_numpy_data(data_params):
    data_path = data_params["data_path"]
    train_data_params = data_params["train_data"]
    test_data_params = data_params["test_data"]
    vali_data_params = data_params["vali_data"]
    X_train, y_train, ps_train, q_train = _load_numpy_data(data_path,train_data_params)
    X_test, y_test, ps_test, q_test     = _load_numpy_data(data_path,test_data_params)
    X_vali, y_vali, ps_vali, q_vali     = _load_numpy_data(data_path,vali_data_params)
    
    train_data = (X_train, y_train, ps_train, q_train)
    test_data  = (X_test , y_test , ps_test , q_test)
    vali_data  = (X_vali , y_vali , ps_vali , q_vali)
    
    return train_data, test_data, vali_data