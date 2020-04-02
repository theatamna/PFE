import numpy as np
import networkx as nx
import os.path as path

def get_dort_graphs(ds_name):
    """
    """
    adjacency_matrices = []
    features_matrices = []
    nodes_in_graph = []
    data_dir = "./data/"
    edges = np.array([(x[0], x[1]) for x in np.loadtxt(data_dir + ds_name + '/' + ds_name + "_A.txt",
                     delimiter = ',', dtype = 'int')])
    graph_labels = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_graph_labels.txt", dtype = 'int')
    graph_labels = (graph_labels > 0) * 1 * graph_labels
    ind = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_graph_indicator.txt", dtype = 'int')
    _, indices = np.unique(ind, return_index = True)

    # Getting node labels
    try:
        nodes_label = np.split(np.loadtxt(data_dir + ds_name + '/' + ds_name + "_node_labels.txt", dtype = 'int'),
                               indices[1:])
    except OSError:
        nodes_label = None

    # Getting adjacency matrices
    n_nodes = len(ind)
    n_graphs = ind[-1]
    G = nx.Graph()
    G.add_nodes_from(range(1, len(ind)))
    G.add_edges_from(edges)
    for i in range(1, n_graphs):
        G_sub = G.subgraph(range(indices[i-1] + 1, indices[i] + 1))
        adjacency_matrices.append(np.array(nx.to_numpy_matrix(G_sub)))
        nodes_in_graph.append(G_sub.number_of_nodes())
    G_sub = G.subgraph(range(indices[-1] + 1, n_nodes + 1))
    adjacency_matrices.append(np.array(nx.to_numpy_matrix(G_sub)))
    nodes_in_graph.append(G_sub.number_of_nodes())

    # Getting feature matrices
    try:
        node_features = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_node_attributes.txt", delimiter = ',')
        for i in range(1, n_graphs):
            features_matrices.append(node_features[indices[i-1] + 1: indices[i] + 1])
        features_matrices.append(node_features[indices[-1] + 1: n_nodes + 1])
    except OSError:
        for num in nodes_in_graph:
            features_matrices.append(np.eye(num))

    return adjacency_matrices, graph_labels, features_matrices, nodes_label
