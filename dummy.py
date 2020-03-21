import torch
from torch.utils.data import TensorDataset
dtype = torch.long

def get_dataset(n_train=32768, n_valid=4096, n_nodes=20, n_classes=2, n_features=3):
    # Generate random adjacency matrices
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    A = A.to(dtype)
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    # Generate train and validation data
    train_y = torch.sum(data[:n_train, :, :], dim=2)
    valid_y = torch.sum(data[n_train:, :, :], dim=2)
    train_y[train_y >= 1] = 1
    valid_y[valid_y >= 1] = 1
    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    train_features = torch.randint(10, (n_train, n_nodes, n_features), dtype=dtype)
    valid_features = torch.randint(10, (n_valid, n_nodes, n_features), dtype=dtype)

    train_features = TensorDataset(train_features)
    valid_features = TensorDataset(valid_features)
    train_data = TensorDataset(data[0], train_y)
    valid_data = TensorDataset(data[1], valid_y)
    return train_data, valid_data, train_features, valid_features
