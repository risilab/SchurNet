import torch
from torch.nn.functional import l1_loss
from torch_geometric.loader import DataLoader
from torch.nn import Module, L1Loss
from tqdm import tqdm
import ptens as p
import wandb
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch.distributed as dist
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy,BinaryAUROC
from typing import Literal
import sys
import os
import typing

from pytorch_optimizer.optimizer.sam import SAM
from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from typing import Callable, Literal, cast, overload, Optional
from torch.optim import Adam, Optimizer
from torch import Tensor


_INF = 1e6

def get_dimension(model):
    for key, value in model.state_dict().items():
        print(f"{key}: {value.shape}")

def deserialize_dimension(checkpoint):
    for key, value in checkpoint['state_dict'].items():
        print(f"{key}: {value.shape}")

loss_arg_type_list = ['MAE','BCEWithLogits','MSE','CrossEntropy']
loss_arg_type = Literal['MAE','BCEWithLogits','MSE','CrossEntropy']
def get_loss_fn(name: loss_arg_type) -> Module:
    return {
        'MAE' : L1Loss,
        'BCEWithLogits' : BCEWithLogitsLoss,
        'MSE' : MSELoss,
        'CrossEntropy' : CrossEntropyLoss,
    }[name]()
def get_score_fn(name):
    return {
        'Binary' : BinaryAccuracy,
        'Multi_class': MulticlassAccuracy,
        'MAE': MeanAbsoluteError,
        'ROC-AUC': BinaryAUROC
    }[name]()

@torch.no_grad()
def test(model: Module,
         dataloader: DataLoader,
         score_fn,
         description: str,
         args,
         device: str,
         position: int = 1,
         rank : int = -1
         ) -> float:
    if rank == -1:
        rank = dist.get_rank()
        print("WARNING: rank is not set. Setting rank to", rank, ", which may result in bug")
    model.eval()
    score_sum = 0
    num_graphs = 0

    print("start training")
    for batch in tqdm(dataloader, desc='Validation', leave=False):
        batch.to(device)
        pred = model(batch)
        score = score_fn(pred, batch.y).detach().item()
        score_sum += score * batch.num_graphs
        num_graphs += batch.num_graphs
    print("training finished")

    # Gather scores from all processes
    score_sum_tensor = torch.tensor([score_sum], dtype=torch.float32, device=device)
    num_graphs_tensor = torch.tensor([num_graphs], dtype=torch.float32, device=device)

    dist.all_reduce(score_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_graphs_tensor, op=dist.ReduceOp.SUM)

    avg_score = score_sum_tensor.item() / num_graphs_tensor.item()
    if rank == 0:
        print(f"Validation Score: {avg_score}")

    model.train()
    return avg_score


class ASAM(SAM):
    def __init__(self, params: PARAMETERS, base_optimizer: OPTIMIZER, rho: float = 0.05, **kwargs):
        super().__init__(params, base_optimizer, rho, True, **kwargs)

    def step(self, closure: Callable[[],Tensor]):#type: ignore
        value = closure()
        super().step(closure)#type: ignore
        return value


def train(
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str,
        args,
        best_val_path,
        checkpoint_path : str = "./checkpoint.tar",
        checkpoint: typing.Dict[str, typing.Any] = None,
        rank : int = -1):
    if rank == -1:
        rank = dist.get_rank()
        print("WARNING: rank is not set. Setting rank to", rank, ", which may result in bug")

    need_update = True # for wandb config update. Only used when we need to update the config due to resume
    loss_fn = get_loss_fn(args.loss)
    score_fn = get_score_fn(args.eval_metric).to(device)
    metric_sgn = 1 if args.eval_metric in ['MAE','MSE'] else -1 #for MAE and MSE, lower is better
    optim = None
    if args.optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(), args.lr)
        print("using Adam as the optimizer")
    elif args.optimizer == 'ASAM':
        optim = ASAM(model.parameters(),Adam,lr=args.lr)
        print("using asam as the optimizer")
    else:
        raise ValueError(f"Optimizer: {args.optimizer} not found")

    # optim = torch.optim.SGD(model.parameters(), args.lr)
    if hasattr(args, 'patience'):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                        'min',
                                                        0.5,
                                                        patience=args.patience)
        print("training with scheduler")

    epoch_start = 0
    best_val = _INF * metric_sgn
    best_val_epoch = epoch_start
    if checkpoint is not None:
        # assert checkpoint['identity'] == wandb.run.id, "checkpoint identity does not match current run. Aborting."
        # check if the state_dict is compatible
        # if the checkpoint state dict keys does not contain module, we need to load the state_dict into the model instead
        print("list(checkpoint['state_dict'].keys())[0]", list(checkpoint['state_dict'].keys())[0])
        if 'module' not in list(checkpoint['state_dict'].keys())[0]:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optim'])
        if hasattr(args, 'patience'):
            sched.load_state_dict(checkpoint['sched'])
        epoch_start = checkpoint['epoch'] + 1
        if hasattr(args, 'best_val') and hasattr (args, 'wandb_resume_id'):
            if metric_sgn == 1:
                best_val = min(checkpoint['best_val'], args.best_val)
            else:
                best_val = max(checkpoint['best_val'], args.best_val)
            print("Resuming from checkpoint with best_val:", best_val)
            print(f"selected from {args.best_val} and {checkpoint['best_val']}")

        print("restarting from epoch:", epoch_start)




    if args.torch_profile and args.nsight_profile:
        raise ValueError("Can't use both torch and nsight profile")

    train_history = []
    val_history = []
    train_scores = []
    lr_history = []

    train_loop = tqdm(range(epoch_start, args.num_epochs), total=args.num_epochs,position=0,file=sys.stdout)

    earlystop_cnt = 0
    earlystop_epoch = args.num_epochs

    # with profile(
    #     activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    #     schedule=schedule(
    #         wait=len(train_dataloader) - 4, # analyze last few steps for each epoch
    #         warmup=1,
    #         active=3,
    #         repeat=3
    #     ),
    #     on_trace_ready=lambda p: trace_handler(p,run_path)
    # ) as p:

    print("start training!")
    for epoch in train_loop:
        # for epoch in range(args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch) # TODO: verify if this works
        loss_sum = 0
        total_graphs = 0
        epoch_loop = tqdm(enumerate(train_dataloader),
                            'train',
                            total=len(train_dataloader),
                            leave=False,
                            position=1,
                            file=sys.stdout)
        for batch_index, batch in epoch_loop:
            batch.to(device)

            def closure():
                optim.zero_grad()
                pred = model(batch)
                y = batch.y
                if args.ds_name == 'ogbg-moltox21':
                    mask = ~torch.isnan(y)
                    pred = pred[mask]
                    y = batch.y[mask]
                loss: torch.Tensor = loss_fn(pred, y)
                loss.backward()
                return loss

            loss = optim.step(closure)

            loss_float: float = loss.detach().item()

            loss_sum += loss_float * batch.num_graphs
            total_graphs += batch.num_graphs
            epoch_loop.set_postfix(avg_loss = loss_sum/total_graphs)

            # if args.wandb and args.log_train_loss_batch:
            #     wandb.log({'train_loss_batch': loss_float})

            # p.step()

        loss_float = loss_sum / total_graphs #average loss
        if hasattr(args, 'patience'):
            sched.step(loss_float,epoch)
        else:
            optim.step(closure)
        print(f"\nepoch: {epoch} loss_float : {loss_float}")

        #calculate scores
        val_score = test(model,val_dataloader,score_fn,'val',args, device, rank=rank)
        if args.log_train_score:
            train_score = test(model,train_dataloader,score_fn,'train',args, device, rank=rank)
            train_scores.append(train_score)

        train_history.append(loss_float)
        val_history.append(val_score)
        train_loop.set_postfix(best_val=best_val,
                                train_loss=loss_float,
                                val=val_score)

        #save checkpoint each epoch
        if val_score * metric_sgn < best_val * metric_sgn:
            best_val = val_score
            best_val_epoch = epoch
            earlystop_cnt = 0 #reset earlystop counter
            #test
            test_score = test(model, test_dataloader,score_fn, 'test_score', args, device, rank=rank)
            if args.wandb and rank == 0:
                wandb.log({
                    "test_score": test_score,
                })
            if rank == 0:
                torch.save(
                    {
                        'state_dict':
                        model.state_dict(),
                        'best_val':
                        best_val,
                        'epoch':
                        epoch,
                        # 'model_state_dict': model.module.state_dict(),
                        'optim':
                        optim.state_dict(),
                        'loss':
                        loss_float,
                        'sched':
                        sched.state_dict()
                        if hasattr(args, 'patience') else None,
                        'test_score':
                        test_score,
                        'identity':
                        wandb.run.id
                    },
                    best_val_path)
                if args.wandb:
                    wandb.save(best_val_path,base_path=os.path.dirname(best_val_path))
        else:
            earlystop_cnt += 1

        if args.need_resume and rank == 0:
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'best_val': best_val,
                    'epoch': epoch,
                    # 'model_state_dict': model.module.state_dict(),
                    'optim': optim.state_dict(),
                    'loss': loss_float,
                    'sched': sched.state_dict() if hasattr(args, 'patience') else None,
                    'best_val': best_val,
                    'identity': wandb.run.id 
                }, checkpoint_path)

            if args.wandb:
                wandb.save(checkpoint_path,base_path=os.path.dirname(checkpoint_path))
                if need_update:
                    wandb.config.update(args, allow_val_change=True)
                    print("wandb config updated")
                    need_update = False

        #early stopping
        current_lr = optim.param_groups[0]['lr']
        lr_history.append(current_lr)
        # if earlystop_cnt > args.earlystop_patience or current_lr < args.min_lr:

        if earlystop_cnt > args.earlystop_patience or (hasattr(args, "min_lr") and current_lr < args.min_lr):
            print(f"Early stop at epoch: {epoch}")
            earlystop_epoch = epoch
            break

        if args.wandb and rank == 0:
            wandb.log({
                'train_loss': loss_float,
                'train_score': train_score if args.log_train_score else None,
                'val_score': val_score,
                'best_val': best_val,
                'lr': current_lr,
            })


    print("\nTraining complete.")

    # load the best model
    # if rank == 0:
    #     state = torch.load(best_val_path)
    #     best_val_epoch: int = state['epoch']
    #     best_val_score: float = state['best_val']
    #     model.load_state_dict(state['state_dict'])

    # train_score = test(model, train_dataloader,score_fn, 'train_score', args, device, rank=rank)
    # test_score = test(model, test_dataloader,score_fn, 'test_score', args, device, rank=rank)

    # print(f"\nBest validation epoch: {best_val_epoch}")
    # print("Scores:")
    # print(f"\tval: {best_val_score}")
    # print(f"\ttest : {test_score}")
    # print(f"\ttrain: {train_score}")
    best_val_score = best_val

    if args.wandb and rank == 0:
        wandb.summary['best_val'] = best_val_score
        wandb.summary['best_val_epoch'] = best_val_epoch
        # wandb.summary['train_score'] = train_score
        wandb.summary['test_score'] = test_score
        wandb.summary['earlystop_epoch'] = earlystop_epoch
        # for clearer recording
        wandb.summary['val_score'] = best_val_score
        wandb.summary['train_loss_batch'] = None

    return model, best_val_epoch, best_val_score, train_history, val_history, lr_history
