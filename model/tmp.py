class Edge_branched_cycle(torch.nn.Module):
    '''do Autobahn operation on branched cycles'''
    def __init__(self, rep_dim, num_channels,use_branched_5_cycles) -> None:
        super().__init__()
        self.br_cycle_mlp = get_mlp((3 + num_channels) * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3) # one more layer mlp
        self.edge_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)

        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.br_cycles = get_branched_6_cycles().values()
        if use_branched_5_cycles:
            self.branched_cycles.append(get_branched_5_cycles().values())
        self.br_cycle_sizes = [get_subgraph_size(S) for S in self.branched_cycles]
        print("br_cycle_sizes: ", self.br_cycle_sizes)
        
        self.br_cycle_autobahns =ModuleList([p.Autobahn(rep_dim,rep_dim * num_channels,br_cycle) 
                                          for br_cycle in self.br_cycles])

    def forward(self,edge_rep,br_cycle_rep,G,data):
        edge2br_cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,br_cycle,min_overlaps=2) 
                       for br_cycle in self.br_cycles]

        edge2br_cycles_aut = [autobahn(edge2br_cycle)
                       for autobahn, edge2br_cycle in zip(self.br_cycle_autobahns,edge2br_cycles)]
        edge2br_cycle_aut = p.batched_subgraphlayer1b.cat(*edge2br_cycles_aut) #nc * num_channels
        edge2br_cycles_linmap = [p.batched_subgraphlayer1b.gather_from_ptensors(edge2br_cycle,G,br_cycle,min_overlaps=size)  #min_overlaps=3 to make sure only cycle to itself
                       for br_cycle,size, edge2br_cycle in zip(self.br_cycles, self.br_cycle_sizes,edge2br_cycles)] #in principle should do linmaps, nc * 4
        edge2br_cycle_linmap = p.batched_subgraphlayer1b.cat(*edge2br_cycles_linmap) #nc * 2
        
        br_cycle_out = self.br_cycle_mlp(torch.cat([br_cycle_rep.torch(),edge2br_cycle_linmap.torch(),edge2br_cycle_aut.torch()],dim=-1)) #nc * 3 -> nc
        br_cycle_out = p.batched_ptensors1b.like(br_cycle_rep,br_cycle_out)

        br_cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(br_cycle_out,G,self.edge,min_overlaps=2)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),br_cycle2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        return edge_out, br_cycle_out