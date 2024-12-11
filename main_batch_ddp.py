'''
to run:
    python main_batch.py experiment_settings.wandb=True
    CUDA_VISIBLE_DEVICES=0 tmux
    cd Autobahn_ptens/
    conda activate pytorch_env
    export CUDA_VISIBLE_DEVICES=1
to debug hydra:
    HYDRA_FULL_ERROR=1 python main_batch.py
to resume
    python main_batch.py +experiment_settings.wandb_resume_id=nc27clic
change config-path:
    python main_batch.py --config-path config_to_run_3_5.18
'''
import os

def set_cuda_visible_devices_from_env():
    local_rank = int(os.environ['LOCAL_RANK'])
    # if local_rank == 0:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # elif local_rank == 1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # elif local_rank == 2:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # else:
    #     raise ValueError("Only 2 GPUs are expected (local_rank should be 0 or 1)")


set_cuda_visible_devices_from_env()

import hydra
from omegaconf import DictConfig
from data_cleaning.utils import load_hyperparameters_from_yaml, model_factory, get_run_from_resume_id
from model.model_utils import visualize_architecture
# from train import train, test
from train_ddp import train, test
import matplotlib.pyplot as plt
from data_cleaning.utils import get_run_path, get_model_size, get_latest_run, get_group
import argparse
import torch
import numpy as np
from data_cleaning.Transforms import PreAddIndex, StandardPreprocessing

import ptens as p
import ptens_base
from data_cleaning.data_loader import get_data_handler
import wandb
import time

import faulthandler

from model.model_utils import fix_seed
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import random
from torch_geometric import seed_everything

def update_run_status(run_id, status, project):
    """ Update the run status in a persistent JSON file.
        run_id is wandb run id
    """
    status_path = f'run_status_{project}.json'
    if os.path.exists(status_path):
        with open(status_path, 'r') as file:
            run_status = json.load(file)
    else:
        run_status = {}

    run_status[run_id] = status
    with open(status_path, 'w') as file:
        json.dump(run_status, file)

@hydra.main(config_path="config",
            config_name="experiments",
            version_base="1.3")
def main(cfg: DictConfig):
    wandb.require("service")

    args = argparse.Namespace()
    args = load_hyperparameters_from_yaml(cfg, args)


    model_kwargs = cfg["model"]
    print("model_kwargs:", model_kwargs)

    if hasattr(args, "wandb_resume_id"):
        print("loading config from specfic run:", args.wandb_resume_id)

        entity = "gnn-explore"
        project = args.wandb_project
        run = get_run_from_resume_id(entity, project, args.wandb_resume_id)
        config = run.config.items()
        summary = run.summary._json_dict


        for k,v in config:
            if k in args:
                # reset the value of the attribute
                print(f"{k} is already in args {getattr(args, k)}")
                if getattr(args, k) != v:
                    print(f"replacing {k} to {v}")
                    setattr(args, k, v)
            else:
                print(f"{k} not in args")
                setattr(args, k, v)

        for k,v in summary.items():
            if k in args:
                # reset the value of the attribute
                print(f"{k} is already in args {getattr(args, k)}")
                if getattr(args, k) != v:
                    print(f"replacing {k} to {v}")
                    setattr(args, k, v)
            else:
                print(f"{k} not in args")
                setattr(args, k, v)

        # convert args namespace to mapping object
        model_kwargs = args.__dict__





    if hasattr(args, 'ptens_buffer'):
        _ptens_buffer = args.ptens_buffer
    # Set random seed for reproducibility
    if not hasattr(args, 'seed'):
        args.seed = 1 

    seed_everything(args.seed) # TODO: add seed to configs
    if args.fixed_seed:
        fix_seed()

    # args.wandb_project = os.environ["WANDB_PROJECT"]

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    #file path

    print("args:", args)

    #load data
    data_handler = get_data_handler(PreAddIndex(),args)

    #set wandb group
    workdir = os.getcwd() # get current work directory
    run_path, run_id = get_run_path(f'{workdir}/runs')
    overview_log_path = f"{run_path}/summary.log"
    args.local_run_id = run_id
    wandb_group = get_group(args, run_id)
    print("workdir: ", workdir)
    print("run_path: ", run_path)


    checkpoint_path = os.path.join(run_path, "checkpoint.tar")

    def running_once(fold_idx, args):


        print("inside running once")
        # Initialize the process group

        dist.init_process_group(backend='nccl')
        # set_cuda_visible_devices()
        # torch.cuda.set_device(dist.get_rank())
        rank = dist.get_rank()
        device = torch.device('cuda:0')

        device_id = rank

        model = model_factory(data_handler, args, **model_kwargs).to(device)

        # ddp_model = DDP(model, device_ids=[device_id])
        # when complains about about find_unused_parameters=TRUE
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # print(model)
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)

        print("ddp successfully initialized")
        best_val_path = os.path.join(run_path, f'best_val_{fold_idx}.tar')
        checkpoint = None
        description = f"{args.ds_name}_{model.__class__.__name__}{'_fix_seed' if args.fixed_seed else ''}_{args.postfix}_{fold_idx}"


        entity = "gnn-explore"
        project = args.wandb_project
        if hasattr(args, "wandb_resume_id"):
            id = args.wandb_resume_id
            print("using wandb resume id", id)
            run = get_run_from_resume_id(entity, project, id)
        if args.wandb and rank == 0:
            print("rank 0, initializing wandb")
            print("has wandb resume id:", hasattr(args, 'wandb_resume_id'))
            if not hasattr(args, 'wandb_resume_id'): # this is the resumed id that wandb will use to resume the run. Note that wandb_resume_id has higher priority than the start run id as it is also used to track whether the current run instance is a resuming instance or not
                if not hasattr(args, "start_run_id"): # this is the run id passed on slurm to intialize a sequence of runs to be resumed
                    id = wandb.util.generate_id()
                    print("generating a new id since no id is provided")
                else:
                    id = args.start_run_id
                    print("using start run id", id)
            # else:
            #     # if run.state == "finished":
            #     #     print("Run has finished. Exiting.")
            #     #     return
            #     if run.state == "running":
            #         print("Run is still active. Exiting")
            #         return
            #     else:
            #         print("Run is not finished. Resuming")
            print("current id:", id)
            # check if the run id has finished. If it has finished successfully, then we do not resume the run and exit
            print("before initializing wandb run")

            run = wandb.init(project=args.wandb_project,entity='gnn-explore', group=wandb_group, name=description,config=args, resume="allow", id=id)
            print("wandb run successfully reinitialized")
        dist.barrier()
        # if wandb.run.resumed:
        if args.wandb and hasattr(args, "wandb_resume_id"):
            print("Resuming run")
            # check if checkpoint exists
            current_count = 0
            if hasattr(args, 'resume_counter'): # resume counter documents the number of times the run has been resumed
                current_count = args.resume_counter
            nonlocal checkpoint_path
            checkpoint_file_name = f"checkpoint_{current_count}.tar" if current_count > 0 else "checkpoint.tar"
            best_val_file_name = f"best_val_{fold_idx}_{current_count}.tar" if current_count > 0 else f"best_val_{fold_idx}.tar"
            # restore using wandb.restore
            wandb_path = os.path.join(entity, project, id)
            print("Restoring checkpoint", wandb.restore(checkpoint_file_name, run_path=wandb_path).name)
            dist.barrier()
            for i in range(8): 
                assert rank <= 8, "more than 8 cards are used here"
                if rank == i:
                    checkpoint = torch.load(wandb.restore(checkpoint_file_name, run_path=wandb_path, replace=True).name)
                dist.barrier()
            dist.barrier()
            current_count += 1
            checkpoint_file_name = f"checkpoint_{current_count}.tar"
            best_val_file_name = f"best_val_{fold_idx}_{current_count}.tar"
            checkpoint_path = os.path.join(run_path, checkpoint_file_name)
            best_val_path = os.path.join(run_path, best_val_file_name)
            args.resume_counter = current_count
            print("checkpoint path:", checkpoint_path)
                # wandb.init(project=args.wandb_project,entity='gnn-explore', group=wandb_group, name=description, config=args, resume=False)

        if rank != 0:
            print("child process, no need to initialize wandb")

        # torch.set_float32_matmul_precision('high')
        # model = Model_EdgeCycle(args.hidden_dim, args.dense_dim, args.out_dim,args.num_layers,
        #                             args.dropout, global_add_pool,args.ds_name,data_handler.ds, device).to(device)



        print(f"Running fold {fold_idx}")
        print("description:", description)
        get_model_size(model)

        # record model size:
        if args.wandb and rank == 0:
            wandb.log({"model_size": get_model_size(model)})

        # if args.visualize_architecture:
        #     visualize_architecture(ddp_model,data_handler.train_dataloader(),device)

        data_handler.set_fold_idx(fold_idx) #get the right split
        # train_loader= data_handler.train_dataloader(distributed=True)
        train_loader= data_handler.train_dataloader(distributed=True)
        val_loader = data_handler.val_dataloader(distributed=True)
        test_loader = data_handler.test_dataloader(distributed=True)

        #train
        # try:
        if args.wandb and rank == 0:
            update_run_status(run.id, 'started', args.wandb_project)
        best_model, best_val_epoch, best_val_score, train_history, val_history, lr_history  = train(
            ddp_model,
            train_loader,
            val_loader,
            test_loader,
            device,
            args=args,
            # best_val_path=f'{run_path}/best_val_{fold_idx}.ckpt',
            best_val_path=best_val_path,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            rank=rank)
        if args.wandb and rank == 0:
            wandb.finish()
        return best_val_score,val_history
        # except Exception as e:
        #     if args.wandb and rank == 0:
        #         wandb.log({'error_message': str(e)})
        #     print("Error in training", str(e))
        # finally:
        #     if args.wandb and rank == 0:
        #         wandb.finish()
        #         update_run_status(run.id, 'completed', args.wandb_project)
 



    start_time = time.time()
    if args.manage_gpu_memory:
        ptens_base.managed_gpu_memory(_ptens_buffer)

    if args.cross_validation:
        best_val_scores = []
        val_histories = np.zeros((args.num_folds,args.num_epochs))
        avg_val_history = np.zeros(args.num_epochs)
        for fold_idx in range(args.num_folds):
            cur_time = time.time()
            args.fold_idx = fold_idx
            best_val_score, val_history = running_once(fold_idx,args)
            best_val_scores.append(best_val_score)
            val_history_array = np.array(val_history)
            padded_val_history = np.pad(val_history_array, (0, args.num_epochs - len(val_history_array)), 'constant',constant_values=val_history[-1])
            val_histories[fold_idx] = padded_val_history
            print(f"Time for one run: {time.time() - cur_time:.1f} seconds")

            ptens_base.clear_managed_gpu_memory()

        try:
            print(f"Average best_val_score: {np.mean(best_val_scores)}")

            avg_val_history = np.mean(val_histories,axis=0)
            best_avg_epoch = np.argmax(avg_val_history)
            best_avg_val = avg_val_history[best_avg_epoch]
            print(f"Best single epoch: {best_avg_epoch + 1}")
            print(f"\tAverage validation value: {best_avg_val}")
            std_val_scores = np.std(val_histories[:,best_avg_epoch])
            print(f"\tStandard deviation: {std_val_scores}")
        except Exception as e:
            print("Error in calculating average")
            raise e
    else:
        running_once(0,args)

    execution_time = time.time() - start_time
    print(f"Experiment time: {execution_time:.1f} seconds")
    dist.destroy_process_group()



if __name__ == '__main__':
    # fault handler:
    faulthandler.enable()
    main()
