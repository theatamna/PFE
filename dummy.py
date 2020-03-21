import torch
from torch.utils.data import TensorDataset
dtype = torch.long

def get_dataset(n_train=65536, n_valid=8192, n_nodes=5, n_classes=2, n_features=3):
    # Generate random adjacency matrices
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    A = A.to(dtype)
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    #data = Normalize_Adj(data) # Normalization
    # Generating labels (whether a node is connected to the second node)
    labels = data[:, 1, :]

    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    labels = torch.split(labels, split_size_or_sections=[n_train, n_valid], dim=0)

    # Generating labels
    train_features = torch.randint(10, (n_train, n_nodes, n_features), dtype=dtype)
    valid_features = torch.randint(10, (n_valid, n_nodes, n_features), dtype=dtype)

    train_features = TensorDataset(train_features)
    valid_features = TensorDataset(valid_features)
    train_data = TensorDataset(data[0], labels[0])
    valid_data = TensorDataset(data[1], labels[1])
    return train_data, valid_data, train_features, valid_features
