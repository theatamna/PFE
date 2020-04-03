import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from get_dort_graphs import *

def prep_dataset(ds_name, train_percentage, batch_size):
    adjacency_matrices, _, features_matrices, nodes_label = get_dort_graphs(ds_name)
    #assert n_train + n_valid == len(adjacency_matrices), "Error: splits must sum exactly to the total num. of data points"
    nb_max_nodes = max(a.shape[0] for a in adjacency_matrices) # Max no. of nodes in a single graph
    d_max = max(x.shape[1] for x in features_matrices) # Max no. of features (different from graph to graph only when node features aren't available)
    n_classes = max(max(x) for x in nodes_label) + 1
    info = [d_max, n_classes] # number of nodes and number of classes

    n_graphs = len(adjacency_matrices)
    n_train = int(n_graphs*train_percentage)
    n_valid = n_graphs - n_train
    # Homogenize dimensions (pad with zeros)
    for i in range(n_graphs):
        A = np.zeros((nb_max_nodes, nb_max_nodes))
        X = np.zeros((nb_max_nodes, d_max))
        A[:adjacency_matrices[i].shape[0], :adjacency_matrices[i].shape[1]] = adjacency_matrices[i]
        adjacency_matrices[i] = A

        X[:features_matrices[i].shape[0], :features_matrices[i].shape[1]] = features_matrices[i]
        features_matrices[i] = X

        y = np.zeros(nb_max_nodes)
        y[:nodes_label[i].shape[0]] = nodes_label[i]
        nodes_label[i] = y

    # Convert all data to make it PyTorch-compatible
    adjacency_matrices = torch.as_tensor(adjacency_matrices)
    features_matrices = torch.as_tensor(features_matrices)
    nodes_label = torch.as_tensor(nodes_label)

    # Split the data
    adjacency_matrices = torch.split(adjacency_matrices, split_size_or_sections=[n_train, n_valid], dim=0)
    features_matrices = torch.split(features_matrices, split_size_or_sections=[n_train, n_valid], dim=0)
    train_y = nodes_label[:n_train]
    valid_y = nodes_label[n_train:n_train+n_valid]

    train_dataset = TensorDataset(adjacency_matrices[0], features_matrices[0], train_y)
    valid_dataset = TensorDataset(adjacency_matrices[1], features_matrices[1], valid_y)

    # Load the data
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    return train_loader, valid_loader, info
