import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

def concat_data(dataset):
    df = pd.DataFrame()
    q = []
    for fold in [1]:
        output_path_1 = f"./data/Fold{fold}/mq2008.{dataset}"
        output_path_2 = f"./data/Fold{fold}/mq2008.{dataset}.group"

        X,y = load_svmlight_file(output_path_1,multilabel=False)
        q += np.loadtxt(output_path_2).astype(int).tolist()
        df_aux = pd.DataFrame(X.toarray(),columns=[str(item) for item in range(X.shape[1])])
        df_aux["label"] = y
        df = pd.concat([df,df_aux],axis=0)
    q_exploded = []
    for i,item in enumerate(q):
        q_exploded += [i]*item
    df["query"] = q_exploded
    return df

def prepare_dataframe(dataset):
    df_ = concat_data(dataset)
    q_df = df_.groupby("query").sum()[["label"]].reset_index()
    allowed_queries = q_df[q_df["label"]!=0]["query"].tolist()
    return df_[df_["query"].isin(allowed_queries)].copy()

def save_numpy_data(X,Y,q,context):
    np.save(f'./data/X_{context}.npy',X)
    np.save(f'./data/y_{context}.npy',Y)
    np.save(f'./data/q_{context}.npy',q,allow_pickle=True)

def save_ranking_dataset_files():
    scaler = StandardScaler()
    df = prepare_dataframe("train")
    X = df.drop(["query","label"],axis=1)
    scaler.fit(X)
    for context in ["train","test","vali"]:
        df = prepare_dataframe(context)
        X = df.drop(["query","label"],axis=1)
        y = df["label"].values
        q = df["query"].values.astype(int)
        X = scaler.transform(X)
        save_numpy_data(X,y,q,context)
        df.to_csv(f"./data/{context}.csv",index=None)