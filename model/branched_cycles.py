import torch
import ptens as p

def get_subgraph_size(S):
    return S.torch().size(0)

def get_branched_6_cycles():
    return {
        'cycle6_one_branch': cycle6_one_branch,
        'cycle6_two_branch_00': cycle6_two_branch_00,
        'cycle6_two_branch_01': cycle6_two_branch_01,
        'cycle6_two_branch_14': cycle6_two_branch_14,
        'cycle6_two_branch_13': cycle6_two_branch_13,
        'cycle6_three_branch_012': cycle6_three_branch_012,
        'cycle6_three_branch_013': cycle6_three_branch_013,
        'cycle6_three_branch_135': cycle6_three_branch_135
    }

def get_branched_5_cycles():
    return {
        'cycle5_one_branch': cycle5_one_branch,
        'cycle5_two_branch_00': cycle5_two_branch_00,
        'cycle5_two_branch_01': cycle5_two_branch_01,
        'cycle5_two_branch_02': cycle5_two_branch_02,
        'cycle5_three_branch_012': cycle5_three_branch_012,
        'cycle5_three_branch_013': cycle5_three_branch_013
    }

#------------------------------cycle5--------------------------------
#cycle5_one_branch
cycle5_one_branch = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    [1, 5, 0, 2, 1, 3, 2, 4, 3, 0, 0]
])
degree = torch.tensor([3, 2, 2, 2, 2, -1], dtype=torch.float)
cycle5_one_branch = p.subgraph.from_edge_index(cycle5_one_branch.float(), degrees=degree)

#cycle5_two_branch_00
cycle5_two_branch_00 = torch.tensor([
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6],
    [1, 5, 6, 0, 2, 1, 3, 2, 4, 3, 0, 0, 0]
])
degree = torch.tensor([4, -1, -1, -1, -1, -1, -1], dtype=torch.float)
cycle5_two_branch_00 = p.subgraph.from_edge_index(cycle5_two_branch_00.float(), degrees=degree)

#cycle5_two_branch_01
cycle5_two_branch_01 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6],
    [1, 5, 0, 2, 6, 1, 3, 2, 4, 3, 0, 0, 1]
])
degree = torch.tensor([3, 3, 2, 2, 2, -1, -1], dtype=torch.float)
cycle5_two_branch_01 = p.subgraph.from_edge_index(cycle5_two_branch_01.float(), degrees=degree)

#cycle5_two_branch_02
cycle5_two_branch_02 = torch.tensor([
    [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6],
    [1, 5, 0, 2, 1, 3, 6, 2, 4, 3, 0, 0, 2]
])
degree = torch.tensor([3, 2, 3, 2, 2, -1, -1], dtype=torch.float)
cycle5_two_branch_02 = p.subgraph.from_edge_index(cycle5_two_branch_02.float(), degrees=degree)

#cycle5_three_branch_012
cycle5_three_branch_012 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
    [1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 3, 0, 0]
])
degree = torch.tensor([3, 3, 3, 2, 2, -1, -1, -1], dtype=torch.float)
cycle5_three_branch_012 = p.subgraph.from_edge_index(cycle5_three_branch_012.float(), degrees=degree)

#cycle5_three_branch_013
cycle5_three_branch_013 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
    [1, 5, 0, 2, 6, 1, 3, 2, 4, 7, 3, 0, 0]
])
degree = torch.tensor([3, 3, 2, 3, 2, -1, -1, -1], dtype=torch.float)
cycle5_three_branch_013 = p.subgraph.from_edge_index(cycle5_three_branch_013.float(), degrees=degree)


#------------------------------cycle6--------------------------------

#define and find 6 cycles with different branches
cycle6_one_branch = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 5, 6, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4, 0]])
degree = torch.tensor([3,2,2,2,2,2,-1],dtype=torch.float)
cycle6_one_branch = p.subgraph.from_edge_index(cycle6_one_branch.float(),degrees=degree)

#cycle6_two_branch_00
cycle6_two_branch_00 = torch.tensor([
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
    [1, 5, 6, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4, 0, 0]])
# degree = torch.tensor([4,2,-1,2,2,2,-1,-1],dtype=torch.float)
degree = torch.tensor([4,-1,-1,-1,-1,-1,-1,-1],dtype=torch.float) #include all template s.t. one vertex has two branches
cycle6_two_branch_00 = p.subgraph.from_edge_index(cycle6_two_branch_00.float(),degrees=degree)

#cycle6_two_branch_01
cycle6_two_branch_01 = torch.tensor([
     [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
    [1, 5, 6, 0, 2, 7, 1, 3, 2, 4, 3, 5, 0, 4, 0, 1]])
degree = torch.tensor([3,3,2,2,2,2,-1,-1],dtype=torch.float)
cycle6_two_branch_01 = p.subgraph.from_edge_index(cycle6_two_branch_01.float(),degrees=degree)

#cycle6_two_branch_14
cycle6_two_branch_14 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7],
    [1, 5, 0, 2, 6, 1, 3, 2, 4, 3, 5, 7, 0, 4, 1, 4]])
degree = torch.tensor([2,3,2,2,3,2,-1,-1],dtype=torch.float)
cycle6_two_branch_14 = p.subgraph.from_edge_index(cycle6_two_branch_14.float(),degrees=degree)

#cycle6_two_branch_13
cycle6_two_branch_13 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7],
    [1, 5, 0, 2, 6, 1, 3, 2, 4, 7, 3, 5, 0, 4, 1, 3]])
degree = torch.tensor([2,3,2,3,2,2,-1,-1],dtype=torch.float)
cycle6_two_branch_13 = p.subgraph.from_edge_index(cycle6_two_branch_13.float(),degrees=degree)

#cycle6_three_branch_012
cycle6_three_branch_012 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    [1, 5, 6, 0, 2, 7, 1, 3, 8, 2, 4, 3, 5, 0, 4, 0, 1, 2]
])
degree = torch.tensor([3, 3, 3, 2, 2, 2, -1, -1, -1], dtype=torch.float)
cycle6_three_branch_012 = p.subgraph.from_edge_index(cycle6_three_branch_012.float(), degrees=degree)

#cycle6_three_branch_013
cycle6_three_branch_013 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    [1, 5, 6, 0, 2, 7, 1, 3, 2, 4, 8, 3, 5, 0, 4, 0, 1, 3]
])
degree = torch.tensor([3, 3, 2, 3, 2, 2, -1, -1, -1], dtype=torch.float)
cycle6_three_branch_013 = p.subgraph.from_edge_index(cycle6_three_branch_013.float(), degrees=degree)

#cycle6_three_branch_135
cycle6_three_branch_135 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 8],
    [1, 5, 0, 2, 6, 1, 3, 2, 4, 7, 3, 5, 0, 4, 8, 1, 3, 5]
])
degree = torch.tensor([2, 3, 2, 3, 2, 3, -1, -1, -1], dtype=torch.float)
cycle6_three_branch_135 = p.subgraph.from_edge_index(cycle6_three_branch_135.float(), degrees=degree)
