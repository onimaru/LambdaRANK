#!/usr/local/bin/python
import datetime
import json
import os
import click
import torch

from torch.utils.tensorboard import SummaryWriter
from lambdarank.data_loader import dataloader, load_numpy_data
from lambdarank.utils import device_loader, count_trainable_parameters
from lambdarank.model_loader import load_model_and_optim
from lambdarank.trainer import train_model
from lambdarank.metrics import eval_model

@click.command()
@click.option('--config', default="configs.json", help='Name of configuration file.')
def main(config):
    ngpu, device = device_loader()
    print(f"Device: {device}")

    # Load configurations
    with open(os.path.join("./configs",config), 'r') as f:
        run_configs = json.load(f)

    writer = SummaryWriter(log_dir=f"runs/{run_configs['run_name']}")
    # Load train and test data
    train_data, test_data, vali_data = load_numpy_data(run_configs["data_parameters"])
    X_train, y_train, ps_train, q_train = train_data
    X_test , y_test , ps_test , q_test  = test_data
    X_vali , y_vali , ps_vali , q_vali  = vali_data

    train_loader = dataloader(
        X_train,
        y_train,
        q_train,
        ps_train
    )
    test_loader = dataloader(
        X_test,
        y_test,
        q_test,
        ps_test
    )
    
    vali_loader = dataloader(
        X_vali,
        y_vali,
        q_vali,
        ps_vali
    )
    
    # Load model and optimizer
    model, optimizer = load_model_and_optim(
        input_dim = train_loader.dataset.width,
        model_configs = run_configs["model_configs"],
        optimizer_configs = run_configs["optimizer_configs"],        
        device = device,
        ngpu = ngpu
    )
    
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
    writer.add_text('Configs', str(run_configs))
    
    # Train model
    model = train_model(
        model,
        optimizer,
        train_loader,
        test_loader,
        run_configs["train_parameters"]["training_epochs"],
        run_configs["train_parameters"]["query_batch_size"],
        run_configs["train_parameters"]["ps_rate"],
        run_configs["train_parameters"]["per_query_sample_size"],
        device,
        writer
    )

    # Final model evaluation
    for k in run_configs["eval_at"]:
        train_result = eval_model(model, train_loader, k, device)
        test_result  = eval_model(model, test_loader , k, device)
        vali_result  = eval_model(model, vali_loader , k, device)
        
        writer.add_scalar(f"overall_nDCG/train", train_result, k)
        writer.add_scalar(f"overall_nDCG/test ", test_result , k)
        writer.add_scalar(f"overall_nDCG/vali ", vali_result , k)
        print(f"ndcg@{k:<4}-> train: {train_result:.4f} | test: {test_result:.4f} | vali: {vali_result:.4f}")
        
    writer.flush()
    writer.close()
    
    # Save model
    # model_name = "_".join(
    #     [run_configs["model_name"],
    #      datetime.datetime.now().strftime("%Y%m%d_%H%M%S")]
    # )
    model_name = run_configs["model_name"]
    torch.save(model.state_dict(), os.path.join(run_configs["models_path"],model_name))
    
if __name__ == '__main__':
     main()