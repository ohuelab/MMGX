######################
### Import Library ###
######################

import numpy as np
import networkx as nx

##########################
### Utilities Function ###
##########################

# nodes and edges to nx
def g_to_nx(nodes, edges):
    G = nx.Graph()

    for n in nodes:
        G.add_node(n,idx=n)
    for e1, e2 in edges:
        G.add_edge(e1, e2)

    return G

# Pair data to nx
def p_to_nx(p):
    G = nx.Graph()
    nodes = np.array(p.x_r, dtype=float)
    node_feature = dict()
    for n in range(len(nodes)):
        G.add_node(n, idx=n)
        node_feature[str(n)] = list(nodes[n])
    nx.set_node_attributes(G, node_feature)    
    edges = np.array(p.edge_attr_r, dtype=float)
    edge_feature = dict()
    for i, (e1, e2) in enumerate(zip(p.edge_index_r[0], p.edge_index_r[1])):
        G.add_edge(e1.item(), e2.item())
        edge_feature[(str(e1), str(e2))] = list(edges[i])
    nx.set_edge_attributes(G, edge_feature)
    return G