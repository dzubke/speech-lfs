# This code is based on the code outlined here: 
# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee


# standard libraries
import argparse
import os
import logging
import math
import random
# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_model_native_loss import CTC_train
from speech.utils.io import write_pickle, load_config


def run(config):

    data_cfg = config["data"]
    log_cfg = config["logger"]
    preproc_cfg = config["preproc"]
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]

    logger=None
    use_log = log_cfg["use_log"]

    use_cuda = torch.cuda.is_available()

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger,
                  start_and_end=data_cfg["start_and_end"])
    train_ldr = loader.make_loader(data_cfg["train_set"],
                        preproc, batch_size, num_workers=data_cfg["num_workers"])
    # Model
    model = CTC_train(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)
    model.cuda() if use_cuda else model.cpu()
    model.train()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr= opt_cfg['start_lr'])
    
    exp_factor = math.log(opt_cfg['end_lr'] / opt_cfg['start_lr']) / (opt_cfg['epochs'] * len(train_ldr))
    lr_lambda = lambda x: math.exp(x * exp_factor)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Make lists to capture the logs
    lr_find_loss = []
    lr_find_lr = []

    first_batch = True

    smoothing = 0.03
    iteration = 0
    avg_loss = 0.0

    for i in range(opt_cfg['epochs']):
        print("epoch {}".format(i))
        tq = tqdm.tqdm(train_ldr)
        if iteration > opt_cfg['max_iterations']:
                break
        
        for batch in tq:
            if iteration > opt_cfg['max_iterations']:
                break
            # Training mode and zero gradients
            optimizer.zero_grad()
            
            # Get outputs to calc loss
            loss = model.native_loss(batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update LR
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            lr_scheduler.step()
            
            # smooth the loss
            loss = loss.item()
            if first_batch:
                lr_find_loss.append(loss)
                first_batch = False
            else:
                avg_loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(avg_loss)
                       
            tq.set_postfix(iter=iteration, avg_loss=avg_loss, loss=loss, lr=lr_step)
            iteration += 1 

    save_dict = {"losses": lr_find_loss, "learning_rates": lr_find_lr}   
    write_pickle(os.path.join(config['save_path'], "loss_lr.pickle"), save_dict)

    plt.plot(lr_find_lr, lr_find_loss)
    plt.xscale('log')
    plt.xlabel('learning_rate')
    plt.ylabel('loss')
    plt.savefig(os.path.join(config['save_path'], "plot.png"))
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Find the learning rate for a model.")
    parser.add_argument("config",
        help="A config file with the training configuration.")
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    run(config)
