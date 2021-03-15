import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations 


class NSFNET(object):

    # initialize parameters
    def __init__(self, weight, num_of_candidates):
        self.NUM_NODES = 14
        self.WT = weight
        self.K = num_of_candidates
        self.G = self.gen_graph()
        self.NUM_EDGES = self.G.number_of_edges()
        self.CANDIDAT = self.candidates()       # CANDIDAT[src][dst] = [ path_1, path_2, ... , path_k ]
        self.ACT_TABLE, self.COM_LINK_TABLE = self.action2path()    # ACT_TABLE[src][dst][action] = [ pri_path, alt_path ] ;  COM_LINK_TABLE[src][dst] = [num_of_comm_node(0-1), num_of_comm_node(0-2), ...]
    
    # generate graph
    def gen_graph(self):
        g = nx.DiGraph()
        nodes = list(range(0,14))
        g.add_nodes_from(nodes)
        
        elist = [(0, 1, {"weight": self.WT}), (0, 2, {"weight": self.WT}), (0, 3, {"weight": self.WT}),
                (1, 0, {"weight": self.WT}), (1, 2, {"weight": self.WT}), (1, 7, {"weight": self.WT}),
                (2, 0, {"weight": self.WT}), (2, 1, {"weight": self.WT}), (2, 5, {"weight": self.WT}),
                (3, 0, {"weight": self.WT}), (3, 4, {"weight": self.WT}), (3, 8, {"weight": self.WT}),
                (4, 3, {"weight": self.WT}), (4, 5, {"weight": self.WT}), (4, 6, {"weight": self.WT}),
                (5, 2, {"weight": self.WT}), (5, 4, {"weight": self.WT}), (5, 12, {"weight": self.WT}), (5, 13, {"weight": self.WT}),
                (6, 4, {"weight": self.WT}), (6, 7, {"weight": self.WT}),
                (7, 1, {"weight": self.WT}), (7, 6, {"weight": self.WT}), (7, 10, {"weight": self.WT}),
                (8, 3, {"weight": self.WT}), (8, 9, {"weight": self.WT}), (8, 11, {"weight": self.WT}),
                (9, 8, {"weight": self.WT}), (9, 10, {"weight": self.WT}), (9, 12, {"weight": self.WT}),
                (10, 7, {"weight": self.WT}), (10, 9, {"weight": self.WT}), (10, 11, {"weight": self.WT}), (10, 13, {"weight": self.WT}),
                (11, 8, {"weight": self.WT}), (11, 10, {"weight": self.WT}), (11, 12, {"weight": self.WT}),
                (12, 5, {"weight": self.WT}), (12, 9, {"weight": self.WT}), (12, 11, {"weight": self.WT}),
                (13, 5, {"weight": self.WT}), (13, 10, {"weight": self.WT})
                ]
        g.add_edges_from(elist)         

        return g

    # get "k" candidate paths for src & dst pairs
    def candidates(self):
        path_dic = np.empty([self.NUM_NODES, self.NUM_NODES], dtype=object)
        for src in range(0, self.NUM_NODES):
            for dst in range(0, self.NUM_NODES):
                paths = list(nx.shortest_simple_paths(self.G, src, dst))
                path_dic[src][dst] = np.array(paths[0:self.K], dtype=object)
        return path_dic
    
    # map actions to pri & alt path combination and shared nodes
    def action2path(self):
        act_table = np.empty([self.NUM_NODES, self.NUM_NODES], dtype=object)
        com_link_table = np.empty([self.NUM_NODES, self.NUM_NODES], dtype=object)

        # map actions to pri & alt comb.
        for src in range(0, self.NUM_NODES):
            for dst in range(0, self.NUM_NODES):
                acts = list(combinations(self.CANDIDAT[src][dst], 2))
                act_table[src][dst] = acts

                # calculate shared nodes of each path
                common = []
                for i, path in enumerate(self.CANDIDAT[src][dst]):
                    path_nodes = set(path)
                    for j, other_path in enumerate(self.CANDIDAT[src][dst]):
                        if i!=j:
                            other_nodes = set(other_path)
                            common.append(len(path_nodes & other_nodes) - 2)
                com_link_table[src][dst] = common

        return act_table, com_link_table

    def link_allock(self, src, dst, bw, act):
        saturated = False
       
        # get pri & alt paths
        pri = self.ACT_TABLE[src][dst][act][0]
        alt = self.ACT_TABLE[src][dst][act][1]

        # substract edge weight by allocated bw
        for i, v in enumerate(pri):
            if i < len(pri)-1: self.G[v][pri[i+1]]['weight'] -= bw

        for i, v in enumerate(alt):
            if i < len(alt)-1: self.G[v][alt[i+1]]['weight'] -= bw

        # Check available capacity
        for (u, v, wt) in self.G.edges.data('weight'):
            if wt <= 0 : saturated = True
        
        return saturated

    def render(self):
        nx.draw(self.G, with_labels=True)

