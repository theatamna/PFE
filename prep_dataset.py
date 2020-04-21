import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from get_dort_graphs import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def prep_dataset(ds_name):
    adjacency_matrices, graph_labels, features_matrices, nodes_label = get_dort_graphs(ds_name)
    nb_max_nodes = max(a.shape[0] for a in adjacency_matrices) # Max no. of nodes in a single graph
    d_max = max(x.shape[1] for x in features_matrices) # Max no. of features (different from graph to graph only when node features aren't available)
    n_node_classes = max(max(x) for x in nodes_label) + 1
    n_graph_classes = max(graph_labels) + 1
    info = [d_max, n_graph_classes, n_node_classes] # Number of nodes and number of classes

    n_graphs = len(adjacency_matrices)

    # Homogenize dimensions (pad with zeros)
    for i in range(n_graphs):
        A = np.zeros((nb_max_nodes, nb_max_nodes))
        X = np.zeros((nb_max_nodes, d_max))

        A[:adjacency_matrices[i].shape[0], :adjacency_matrices[i].shape[1]] = adjacency_matrices[i]
        adjacency_matrices[i] = A

        X[:features_matrices[i].shape[0], :features_matrices[i].shape[1]] = features_matrices[i]
        features_matrices[i] = X

    # Convert all data to make it PyTorch-compatible
    adjacency_matrices = torch.as_tensor(adjacency_matrices)
    features_matrices = torch.as_tensor(features_matrices)
    graph_labels = torch.as_tensor(graph_labels)
    return adjacency_matrices, features_matrices, graph_labels, info

def get_folded_data(ds_name, batch_size, n_folds):
    """
    Splits data into K-folds for cross-validation
    inputs:
    ds_name: String, name of the dataset
    batch_size: Int,
    n_folds: Int, number of folds
    Returns:
    folded_train_data: A list of dataloaders for each "training data fold"
    folded_test_data: A list of dataloaders for each "test data fold"
    info: [max_numbers_of_features, number_of_classes]
    """
    adj, feat, labels, info = prep_dataset(ds_name)
    kf = KFold(n_splits=n_folds, shuffle=True)
    folded_train_data = []
    folded_test_data = []

    for train_index, test_index in kf.split(adj):
        train_fold = TensorDataset(adj[train_index], feat[train_index], labels[train_index])
        test_fold = TensorDataset(adj[test_index], feat[test_index], labels[test_index])

        folded_train_data.append(DataLoader(dataset=train_fold,
                                           batch_size=batch_size,
                                           shuffle=True))

        folded_test_data.append(DataLoader(dataset=test_fold,
                                           batch_size=batch_size,
                                           shuffle=False))
    return folded_train_data, folded_test_data, info
