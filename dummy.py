import torch
from torch.utils.data import TensorDataset
dtype = torch.float

def Normalize_Adj(A):
    A_tilda = A + torch.eye(A.shape[1]).repeat(A.shape[0], 1, 1)
    D_tilda = torch.diag_embed(torch.sum(A_tilda, 2).pow(-0.5))
    A_hat = D_tilda.bmm(A_tilda).bmm(D_tilda)
    return A_hat

def get_dataset(n_train=65536, n_valid=8192, n_nodes=n_nodes, n_classes=n_classes):
    # Generate random adjacency matrices
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    A = A.to(dtype)
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    data = Normalize_Adj(data) # Normalization
    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    # Generating labels
    train_y = torch.randint(n_classes, (n_train, n_nodes), dtype=dtype)
    valid_y = torch.randint(n_classes, (n_valid, n_nodes), dtype=dtype)

    train_data = TensorDataset(data[0], train_y)
    valid_data = TensorDataset(data[1], valid_y)
    return train_data, valid_data