# compability methods
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import argparse
import os
import itertools
import json
import logging
import math
import random
import time
# third-party libraries
#import apex
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import tqdm
import yaml
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_model_train import CTC_train
from speech.utils.io import load_config, load_from_trained, read_pickle, save, write_pickle 
from speech.utils.model_debug import check_nan_params_grads, log_model_grads, plot_grad_flow_line, plot_grad_flow_bar
from speech.utils.model_debug import save_batch_log_stats, log_batchnorm_mean_std, log_param_grad_norms
from speech.utils.model_debug import get_logger_filename, log_cpu_mem_disk_usage


BLANK_IDX = 0


def run_epoch(model, 
              optimizer, 
              train_ldr, 
              logger, 
              debug_mode:bool, 
              tbX_writer, 
              iter_count:int, 
              avg_loss:float, 
              local_rank:int, 
              loss_name:str, 
              save_path:str,
              chkpt_per_epoch:int):
    """
    Performs a forwards and backward pass through the model
    Args:
        iter_count (int): count of iterations
        save_path (str): path to directory where model is saved
        chkpt_per_epoch (int): # checkpoints for each epoch (including at the end of the epoch) that will be saved
    """
    is_rank_0 = (torch.distributed.get_rank() == 0 )
    use_log = (logger is not None) and is_rank_0
    model_t, data_t = 0.0, 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)  if is_rank_0 else train_ldr
    log_modulus = 100     # limits certain logging function to report less frequently
    exp_w = 0.985        # exponential weight for exponential moving average loss        
    avg_grad_norm = 0
    # batch_counter is use to checkpoint the model after going through 50% of the dataset
    batch_counter = 0
    device = torch.device("cuda:" + str(local_rank))
    print(f"rank {local_rank}: device: {device}")
    print(f"rank {local_rank}: loader length", len(train_ldr))

    # model compatibility for using multiple gpu's 
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)): #, apex.parallel.DistributedDataParallel)): 
       model_module = model.module 
    else: 
        model_module = model

    print(f"rank {local_rank}:tq type:", type(tq))
    #train_iter = iter(train_ldr)
    print(f"rank {local_rank}: after iterator assigned")
    #print(f"first batch: {next(iter(train_ldr))}")
    #print(f"first batch: {next(iter(train_ldr))}")
    #print(f"first batch: {next(iter(train_ldr))}")
    for batch in tq:
        if use_log: logger.info(f"train: ====== Iteration: {iter_count} in run_epoch =======")
        #print(f"rank {local_rank}: inside loop")
        
        ##############  Mid-epoch checkpoint ###############
        if is_rank_0 and batch_counter % (len(train_ldr) // chkpt_per_epoch) == 0 and batch_counter != 0:
            preproc = train_ldr.dataset.preproc
            save(model_module, preproc, save_path, tag='ckpt')
        batch_counter += 1
        ####################################################

        #print(f"rank {local_rank}: loop begins")
        batch = list(batch)    # this was added as the batch generator was being exhausted when it was called
        #print(f"rank {local_rank}: after temp batch")
        if use_log: 
            if debug_mode:  
                save_batch_log_stats(batch, logger)
                log_batchnorm_mean_std(model_module.state_dict(), logger)
 
        start_t = time.time()
        optimizer.zero_grad()
        if use_log: logger.info(f"train: Optimizer zero_grad")
        #print(f'rank {local_rank}: after optimizer zero_grad')
        # calcuating the loss outside of model.loss to allow multi-gpu use
        inputs, labels, input_lens, label_lens = model_module.collate(*batch)
        #print(f'rank {local_rank}: after collate')
        inputs = inputs.cuda() #.to(device) #local_rank)
        #print(f'rank {local_rank}: after inputs to cuda')
        out, rnn_args = model(inputs, softmax=False)
        #print(f'rank {local_rank}: after model inference')

        if loss_name == "native":
            loss = native_loss(out, labels, input_lens, label_lens, BLANK_IDX)
        elif loss_name == "awni":
            loss = awni_loss(out, labels, input_lens, label_lens, BLANK_IDX)
        elif loss_name == "naren":
            loss =naren_loss(out, labels, input_lens, label_lens, BLANK_IDX)
        #print(f"rank {local_rank}: after loss calc")
        
        if use_log: logger.info(f"train: Loss calculated")
    
        ############# amp change ##################################################################
        #with apex.amp.scale_loss(loss, optimizer) as scaled_loss:                                      #
        #   scaled_loss.backward()                                                                #

        ############# non-amp change  #############################################################
        loss.backward()                                                                           #
        #print(f"rank {local_rank}: after backwards call")
        #print("loss", loss)
        if use_log: logger.info(f"train: Backward run ")
        if use_log: 
            if debug_mode: 
                plot_grad_flow_bar(model_module.named_parameters(),  get_logger_filename(logger))
                log_param_grad_norms(model_module.named_parameters(), logger)

        ############# amp change ##################################################################
        #grad_norm = nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer), 200).item()            #
        
        ############# non-amp change ##################################################################
        grad_norm = nn.utils.clip_grad_norm_(model_module.parameters(), 200) 
        #grad_norm = 0
        #print(f"rank {local_rank}: after grad_norm")
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        if use_log: logger.info(f"train: Grad_norm clipped ")
        #print(f"rank {local_rank}: before optimizer step")
        optimizer.step()
        if use_log: logger.info(f"train: Optimizer step taken")
        #print(f"rank {local_rank}: after optimizer step")
        if is_rank_0:  # logging on rank_0 process
            #print("inside rank 0 logging")
            loss = loss.item()
            if use_log: logger.info(f"train: loss reassigned ")

            prev_end_t = end_t
            end_t = time.time()
            model_t += end_t - start_t
            data_t += start_t - prev_end_t
            if use_log: logger.info(f"train: time calculated ")
            #print("after time calc")
            if iter_count == 0:
                avg_loss = loss
                avg_grad_norm = grad_norm
            else: 
                avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
                avg_grad_norm = exp_w * avg_grad_norm + (1 - exp_w) * grad_norm
            if use_log: logger.info(f"train: Avg loss: {avg_loss}")
            #print("avg_loss, grad_norm calc")
            tbX_writer.add_scalars('train/loss', {"loss": loss}, iter_count)
            tbX_writer.add_scalars('train/loss', {"avg_loss": avg_loss}, iter_count)
            tbX_writer.add_scalars('train/grad', {"grad_norm": avg_grad_norm}, iter_count)
            tq.set_postfix(iter=iter_count, loss=loss, 
                avg_loss=avg_loss, grad_norm=grad_norm,
                model_time=model_t, data_time=data_t)
            #print("tq and tbX writing")
            if use_log: logger.info(f'train: loss is inf: {loss == float("inf")}')
            if use_log: logger.info(f"train: iter={iter_count}, loss={round(loss,3)}, grad_norm={round(grad_norm,3)}")
        
            if iter_count % log_modulus == 0:
                if use_log: log_cpu_mem_disk_usage(logger)
        
        if check_nan_params_grads(model_module.parameters()):
            if use_log:
                logger.error(f"train: labels: {[labels]}, label_lens: {label_lens} state_dict: {model_module.state_dict()}")
                log_model_grads(model_module.named_parameters(), logger)
                save_batch_log_stats(batch, logger)
                log_param_grad_norms(model_module.named_parameters(), logger)
                plot_grad_flow_bar(model_module.named_parameters(), get_logger_filename(logger))
            debug_mode = True
            torch.autograd.set_detect_anomaly(True)

        #print(f"rank {local_rank}: loop end")
        iter_count += 1

    return iter_count, avg_loss

def native_loss(out, labels, input_lens, label_lens, blank_idx):
    """
    """
    ############## Native loss code ############################################################
    log_probs = nn.functional.log_softmax(out, dim=2)                                        # 
    loss_fn = torch.nn.CTCLoss(blank=blank_idx, reduction='sum', zero_infinity=True)         #     
    loss = loss_fn(log_probs.permute(1,0,2).float(), labels, input_lens, label_lens)         #
    return loss


def awni_loss(out, labels, input_lens, label_lens, blank_idx):
    import functions.ctc as ctc #awni hannun's ctc bindings
    ############## Awni loss code  #############################################################
    loss_fn = ctc.CTCLoss(blank_label=blank_idx)   
    loss = loss_fn(out, labels, input_lens, label_lens)      
    return loss

def naren_loss(out, labels, input_lens, label_lens, blank_idx):
    from warpctc_pytorch import CTCLoss
    loss_fn = CTCLoss(blank=blank_idx, size_average=True, length_average=False)
    out = out.permute(1,0,2).float().cpu() #permuation for sean ctc
    #print(f"out shape after permute: {float_out.size()}, sum: {torch.sum(float_out)}")
    loss = loss_fn(out, labels, input_lens, label_lens)
    return loss



def eval_dev(model, ldr, preproc,  logger, loss_name):
    losses = []; all_preds = []; all_labels = []
        
    model.set_eval()
    preproc.set_eval()  # this turns off dataset augmentation
    use_log = (logger is not None)
    if use_log: logger.info(f"eval_dev: set_eval ")


    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            if use_log: logger.info(f"eval_dev: =====Inside batch loop=====")
            batch = list(batch)
            if use_log: logger.info(f"eval_dev: batch converted")
            preds = model.infer(batch)
            if use_log: logger.info(f"eval_dev: infer call")
            
            inputs, labels, input_lens, label_lens = model.collate(*batch)
            inputs = inputs.cuda()
            out, rnn_args = model(inputs, softmax=False)


            if loss_name == "native":
                loss = native_loss(out, labels, input_lens, label_lens, BLANK_IDX)
            elif loss_name == "awni":
                loss = awni_loss(out, labels, input_lens, label_lens, BLANK_IDX)
            elif loss_name == "naren":
                loss = naren_loss(out, labels, input_lens, label_lens, BLANK_IDX)

            if use_log: logger.info(f"eval_dev: loss calculated as: {loss.item():0.3f}")
            if use_log: logger.info(f"eval_dev: loss is nan: {math.isnan(loss.item())}")
            losses.append(loss.item())
            if use_log: logger.info(f"eval_dev: loss appended")
            #losses.append(loss.data[0])
            all_preds.extend(preds)
            if use_log: logger.info(f"eval_dev: preds: {preds}")
            all_labels.extend(batch[1])        #add the labels in the batch object
            if use_log: logger.info(f"eval_dev: labels: {batch[1]}")

    model.set_train()
    preproc.set_train()
    if use_log: logger.info(f"eval_dev: set_train")

    loss = sum(losses) / len(losses)
    if use_log: logger.info(f"eval_dev: Avg loss: {loss}")

    results = [(preproc.decode(l), preproc.decode(p))              # decodes back to phoneme labels
               for l, p in zip(all_labels, all_preds)]
    if use_log: logger.info(f"eval_dev: results {results}")
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    if use_log: logger.info(f"CER: {cer}")

    return loss, cer

def run(local_rank, config):

    data_cfg = config["data"]
    log_cfg = config["logger"]
    preproc_cfg = config["preproc"]
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    train_cfg = config['training']    

    # setting up the distributed training environment
    if train_cfg['distributed']:
        dist.init_process_group(                                   
            backend='nccl'
            #init_method='env://',
        )
        torch.cuda.set_device(local_rank)
        print(f"local_rank: {local_rank}, dist.get_rank: {torch.distributed.get_rank()}")
        is_rank_0 = (torch.distributed.get_rank() == 0)
    else:
        is_rank_0 = True

    use_log = log_cfg["use_log"] and is_rank_0

    debug_mode = log_cfg["debug_mode"]    
    if debug_mode: torch.autograd.set_detect_anomaly(True)

    # TODO, drz, replace with get_logger function
    if use_log:
        # create logger
        logger = logging.getLogger("train_log")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_cfg["log_file"])
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger = None
    

    if is_rank_0: # will create tensorboard X writer if rank 0 process
        tbX_writer = SummaryWriter(logdir=config["save_path"])
    else:
        tbX_writer = None
    
    # Load previous train state: dict with contents:
    #   {'start_epoch': int, 'run_state': (int, float), 'best_so_far': float, 'learning_rate': float}
    train_state_path = os.path.join(config["save_path"], "train_state.pickle")
    if os.path.exists(train_state_path):
        print(f"load train_state from: {train_state_path}")
        train_state = read_pickle(train_state_path)
    else:   # if train_path doesn't exist, create empty dict to load from config
        print(f"load train_state from config")
        train_state = dict()
    # the get-statements will load from train_state if key exists, and from opt_cfg otherwise
    run_state = train_state.get('run_state',  opt_cfg['run_state'])
    best_so_far = train_state.get('best_so_far', opt_cfg['best_so_far'])
    start_epoch =  train_state.get('start_epoch', # train_state used to use 'next_epoch' 
                                    train_state.get('next_epoch', opt_cfg['start_epoch'])
    )
    learning_rate = opt_cfg['learning_rate']

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger, 
                  start_and_end=data_cfg["start_and_end"])
    
    if train_cfg['distributed']:
        train_ldr = loader.make_ddp_loader(data_cfg["train_set"], preproc, batch_size, num_workers=data_cfg["num_workers"])
    else: 
        train_ldr = loader.make_loader(data_cfg["train_set"], preproc, batch_size, num_workers=data_cfg["num_workers"])  


    dev_ldr_dict = dict() # dict that includes all the dev_loaders
    if is_rank_0:       # evaluation will only be done in rank_0 process
        for dev_name, dev_path in data_cfg["dev_sets"].items():
            # data_cfg["num_workers"] = 0 if 'distributed' is tru
            dev_ldr = loader.make_loader(dev_path, preproc, batch_size=8, num_workers=data_cfg["num_workers"])
            dev_ldr_dict.update({dev_name: dev_ldr})

    # Model
    model = CTC_train(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)
    if model_cfg["load_trained"]:
        model = load_from_trained(model, model_cfg)
        print(f"Succesfully loaded weights from trained model: {model_cfg['trained_path']}")
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=learning_rate,   # from train_state or opt_config
                    momentum=opt_cfg["momentum"],
                    dampening=opt_cfg["dampening"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=opt_cfg["sched_step"], 
        gamma=opt_cfg["sched_gamma"])

    loss_name = train_cfg['loss_name']

    # multi-gpu training
    if train_cfg["multi_gpu"]:
        assert torch.cuda.device_count() > 1, "multi_gpu selected but less than on GPU available"
        model = torch.nn.DataParallel(model)
        model_module = model.module
        model.cuda() if use_cuda else model.cpu()
    
    elif train_cfg['distributed']:
        model.cuda(local_rank)
    
        #if train_cfg['apex']:
        #    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=train_cfg['opt_level']) 
        #    model = apex.parallel.DistributedDataParallel(model)
        #else:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        
        model_module = model.module
    else:
        # allows for compatbility with data-parallel models
        model_module = model
        model.cuda() if use_cuda else model.cpu()

    if use_log: 
        logger.info(f"train: ====== Model, loaders, optimimzer created =======")
        logger.info(f"train: model: {model}")
        logger.info(f"train: preproc: {preproc}")
        logger.info(f"train: optimizer: {optimizer}")
        logger.info(f"train: config: {config}")

    # printing to the output file
    if is_rank_0:
        print(f"====== Model, loaders, optimimzer created =======")
        print(f"model: {model}")
        print(f"preproc: {preproc}")
        print(f"optimizer: {optimizer}")
        print(f"config: {config}")

    for epoch in range(start_epoch, opt_cfg["epochs"]):
        if use_log: logger.info(f"Starting epoch: {epoch}")
        start = time.time()
        for group in optimizer.param_groups:
            if is_rank_0: print(f'learning rate: {group["lr"]}')
            if use_log: logger.info(f"train: learning rate: {group['lr']}")
        

        try:
            run_state = run_epoch(model, optimizer, train_ldr, logger, debug_mode, tbX_writer, 
                                    *run_state, local_rank, loss_name, config['save_path'],
                                    train_cfg['checkpoints_per_epoch'])
        except Exception as err:
            if use_log: 
                logger.error(f"Exception raised: {err}")
                logger.error(f"train: ====In except block====")
                logger.error(f"train: state_dict: {model_module.state_dict()}")
                log_model_grads(model_module.named_parameters(), logger)
            raise Exception('Failure in run_epoch').with_traceback(err.__traceback__)
        finally: # used to ensure that plots are closed even if exception raised
            plt.close('all')
    
        # update the learning rate
        lr_scheduler.step()       
 
        if use_log:
            logger.info(f"train: ====== Run_state finished =======") 
            logger.info(f"train: preproc type: {type(preproc)}")
        if is_rank_0:
            msg = "Epoch {} completed in {:.2f} (hr)."
            epoch_time_hr = (time.time() - start)/60/60
            print(msg.format(epoch, epoch_time_hr))
            if use_log: logger.info(msg.format(epoch, epoch_time_hr))
            tbX_writer.add_scalars('train/stats', {"epoch_time_hr": epoch_time_hr}, epoch)
    
            # the logger needs to be removed to save the model
            if use_log: preproc.logger = None
            speech.save(model_module, preproc, config["save_path"])
            if use_log: logger.info(f"train: ====== model saved =======")
            if use_log: preproc.logger = logger

            # creating the dictionaries that hold the PER and loss values
            dev_loss_dict = dict()
            dev_per_dict = dict()
            # iterating through the dev-set loaders to calculate the PER/loss
            for dev_name, dev_ldr in dev_ldr_dict.items():
                print(f"evaluating devset: {dev_name}")
                if use_log: logger.info(f"train: === evaluating devset: {dev_name} ==")
                dev_loss, dev_per = eval_dev(model_module, dev_ldr, preproc, logger, loss_name)

                dev_loss_dict.update({dev_name: dev_loss})
                dev_per_dict.update({dev_name: dev_per})

                if use_log: logger.info(f"train: ====== eval_dev {dev_name} finished =======")
                
                # Save the best model on the dev set
                if dev_name == data_cfg['dev_set_save_reference']:
                    print(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                    
                    if use_log: logger.info(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                    if dev_per < best_so_far:
                        if use_log: preproc.logger = None   # remove the logger to save the model
                        best_so_far = dev_per
                        speech.save(model_module, preproc,
                                config["save_path"], tag="best")
                        if use_log: 
                            preproc.logger = logger
                            logger.info(f"model saved based per on: {dev_name} dataset")

                        print(f"UPDATED: best_model based on PER {best_so_far} for {dev_name} devset")
                
            per_diff_dict = calc_per_difference(dev_per_dict) 

            tbX_writer.add_scalars('dev/loss', dev_loss_dict, epoch)
            tbX_writer.add_scalars('dev/per', dev_per_dict, epoch)
            tbX_writer.add_scalars('dev/per/diff', per_diff_dict, epoch)

            learning_rate = list(optimizer.param_groups)[0]["lr"]
            # save the current state of training
            train_state = {"start_epoch": epoch + 1, 
                           "run_state": run_state, 
                           "best_so_far": best_so_far,
                           "learning_rate": learning_rate}
            write_pickle(os.path.join(config["save_path"], "train_state.pickle"), train_state)


def calc_per_difference(dev_per_dict:dict) -> dict:
    """
    Calculates the differecence between the speak testset PER and the training-dev sets. This
    difference is a measure of data mismatch.
    """
    per_diff_dict = dict()

    for name, per in dev_per_dict.items():
        if not name=='speak':
            diff_name = name + "-speak"
            per_diff_dict[diff_name] = dev_per_dict.get('speak', 0.0) - dev_per_dict.get(name, 0.0)
    
    return per_diff_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")
    parser.add_argument("config",
        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    #parser.add_argument('--rank', default=0, type=int,
    #                    help='ranking within the compute nodes')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for singe node, aka gpu index.')
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False

    train_cfg = config['training']    

    os.environ['OMP_NUM_THREADS'] = str(train_cfg['OMP_NUM_THREADS'])

    if train_cfg['distributed'] and train_cfg['use_spawn']:
        mp.spawn(run, nprocs=gpu_per_node, args=(config, ))
    elif train_cfg['distributed'] and not train_cfg['use_spawn']:
        run(local_rank=args.local_rank, config=config)
    else:
        run(local_rank=0, config=config)
