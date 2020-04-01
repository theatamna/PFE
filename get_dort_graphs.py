import numpy as np
import networkx as nx
import os.path as path

def get_dort_graphs(ds_name):
    "WORK IN PROGRESS"
    adjacency_matrices = []
    features_matrices = []
    data_dir = "./data/"
    edges = np.array([(x[0], x[1]) for x in np.loadtxt(data_dir + ds_name + '/' + ds_name + "_A.txt",
                     delimiter = ',', dtype = 'int')])
    graph_labels = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_graph_labels.txt")
    graph_labels = (graph_labels > 0) * 1
    ind = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_graph_indicator.txt", dtype = 'int')
    _, indices = np.unique(ind, return_index = True)
    #
    n_nodes = len(ind)
    n_graphs = ind[-1]
    G = nx.Graph()
    G.add_nodes_from(range(1, len(ind)))
    G.add_edges_from(edges)
    for i in range(1, n_graphs):
        G_sub = G.subgraph(range(indices[i-1] + 1, indices[i] + 1))
        adjacency_matrices.append(np.array(nx.to_numpy_matrix(G_sub)))
    G_sub = G.subgraph(range(indices[-1] + 1, n_nodes + 1))
    adjacency_matrices.append(np.array(nx.to_numpy_matrix(G_sub)))
    return adjacency_matrices, graph_labels




ds_name = "PTC_MR"
K = get_dort_graphs(ds_name)
