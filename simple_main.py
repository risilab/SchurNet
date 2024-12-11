from argparse import Namespace
from data_cleaning.data_loader import get_data_handler
from data_cleaning.Transforms import PreAddIndex, StandardPreprocessing
from data_cleaning.utils import get_model_size
from train import get_loss_fn
from torch_geometric.nn import global_add_pool, global_mean_pool
from model.Model_GIN import Model_GIN
from model.Model_NodeEdge_batch import Model_NodeEdge_batch
from model.Model_NodeEdge import Model_NodeEdge
from model.Model_NodeEdge_0th_batch import Model_NodeEdge_0th_batch
from model.Model_GIN_batch import Model_GIN_batch
from model.GIN import GIN
from model.Model_EdgeCycle_0th_batch import Model_EdgeCycle_0th_batch
from model.Model_EdgeCycle_0th_batch_small import Model_EdgeCycle_0th_batch_small
from Autobahn_ptens.model.Model_EdgeCycle_Aut_0th_batch import Model_EdgeCycle_Aut_0th_batch
from Autobahn_ptens.model.Model_EdgeCycle_Aut_0th_batch_old import Model_EdgeCycle_Aut_0th_batch_old
from model.Model_EdgeCycle_Aut_0th_batch_on_edge2cycle import Model_EdgeCycle_Aut_0th_batch_on_edge2cycle
from model.Model_EdgeCycle_batch import Model_EdgeCycle_batch
from model.Model_EdgeCycle import Model_EdgeCycle
from model.Model_EdgeCycle_old import Model_EdgeCycle_old
from model.Model_ZINC_Ext_batch import Model_ZINC_Ext_batch
# from model.Model_EdgeCycle_Aut_0th_batch_on_edge2cycle_small import Model_EdgeCycle_Aut_0th_batch_on_edge2cycle_small
from model.Model_EdgeCycle_pLinear_0th_batch_on_edge2cycle_small import Model_EdgeCycle_pLinear_0th_batch_on_edge2cycle_small
from model.Model_EdgeCycle_Aut_on_branch_0th_batch_small import Model_EdgeCycle_Aut_on_branch_0th_batch_small
from model.ablation_Model_Aut_on_branch_small import ablation_Model_Aut_on_branch_small
from model.ablation_Model_Aut_on_cycle_only_small import ablation_Model_Aut_on_cycle_only_small
from model.final_Model_Aut_on_edge2cycle_small import final_Model_Aut_on_edge2cycle_small
from model.final_Model_Aut_on_edge2cycle_small_ogb import final_Model_Aut_on_edge2cycle_small_ogb
from model.Model_EdgeCycle_Aut_double_0th_batch import Model_EdgeCycle_Aut_double_0th_batch
from model.Model_EdgeCycle_Aut_double_0th_batch_2 import Model_EdgeCycle_Aut_double_0th_batch_2
from model.ablation_Model_Aut_on_branch import ablation_Model_Aut_on_branch

import torch
import matplotlib.pyplot as plt
from model.model_utils import fix_seed
import ptens as p

def fix_weight(model, w):
    '''fix the weight of the model to scalar w'''
    for name, param in model.named_parameters():
        param.data.fill_(w)
        
# ---- Main --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("test")
    device = 'cuda:0'

    args = Namespace(
        ds_name='ZINC',
        batch_size=2,
        eval_batch_size=10,
        num_folds=10,
        fold_idx = 0,
        lr = 0.01,
        cycle_sizes = [5,6]
    )
    fix_seed(0)
    data_handler = get_data_handler(PreAddIndex(),args)
    train_loader = data_handler.train_dataloader()
    
    model = ablation_Model_Aut_on_branch(rep_dim=4,dense_dim=4,out_dim=1,num_layers=1,dropout=0.,
                      readout=global_add_pool,num_channels=2,cycle_sizes=args.cycle_sizes,use_branched_5_cycles=True,gather_overlap=2,
                      ds_name=args.ds_name,dataset=data_handler.ds).to(device)

    # model = final_Model_Aut_on_edge2cycle_small_ogb(rep_dim=4,dense_dim=4,out_dim=1,num_layers=1,dropout=0.5,
    #                   readout=global_add_pool,num_channels=2,cycle_sizes=args.cycle_sizes,gather_overlap=1,
    #                   ds_name=args.ds_name,dataset=data_handler.ds).to(device)
    # model = Model_GIN_batch(hidden_dim=4,dense_dim=4,out_dim=1,num_layers=1,dropout=0.,
    #                   readout=global_add_pool,ds_name=args.ds_name,dataset=data_handler.ds).to(device)
    # model = Model_EdgeCycle_0th_batch(rep_dim=4,dense_dim=4,out_dim=1,num_layers=2,dropout=0.,
    #                   readout=global_add_pool,cycle_sizes=args.cycle_sizes,gather_overlap=2,
    #                   ds_name=args.ds_name,dataset=data_handler.ds).to(device)
    # fix_weight(model,0.1)
    
    optim = torch.optim.SGD(model.parameters(), args.lr)
      
    loss_fn = get_loss_fn('MAE')
    loss_sum = 0
    num_step = 0
    train_loss = []

    for epoch in range(2):
        num_step = 0
        for data in train_loader:
            print("batch is:", data)
            # print("data.y:", data.y)
            # print("edge_attr:", data.edge_attr)
            data.to(device)
            pred = model(data)
            # print("pred : ",pred)
            loss = loss_fn(pred,data.y)
            print(f"loss: {loss}")
            loss.backward()
            optim.step()
            
            train_loss.append(loss.detach().item())
            # print(p.subgraph.cached())
            num_step += 1
            if num_step >= 5:
                break
        
        
        print(f"loss_sum: {loss_sum}")
        print(f"finished epoch: {epoch}")

#plot training loss
plt.plot(train_loss)
plt.legend(['train_loss'])
plt.xlabel('step')
plt.ylabel('Loss')
plt.savefig(f'output/loss.png')
