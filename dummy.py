import torch
from torch.utils.data import TensorDataset
dtype = torch.long

def get_dataset(n_train=65536, n_valid=8192, n_nodes=5, n_classes=4, n_features=3):
    # Generate random adjacency matrices
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    A = A.to(dtype)
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    #data = Normalize_Adj(data) # Normalization
    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    # Generating labels
    train_y = torch.randint(n_classes, (n_train, n_nodes), dtype=dtype)
    valid_y = torch.randint(n_classes, (n_valid, n_nodes), dtype=dtype)

    train_features = torch.randint(10, (n_train, n_nodes, n_features), dtype=dtype)
    valid_features = torch.randint(10, (n_valid, n_nodes, n_features), dtype=dtype)

    train_features = TensorDataset(train_features)
    valid_features = TensorDataset(valid_features)
    train_data = TensorDataset(data[0], train_y)
    valid_data = TensorDataset(data[1], valid_y)
    return train_data, valid_data, train_features, valid_features
