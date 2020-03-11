import torch
from torch.utils.data import TensorDataset
dtype = torch.float

def Normalize_Adj(A): # may need to use torch.repeat and torch.bmm
    n = A.shape[0]
    A_tilda = A + torch.eye(A.shape[1]).repeat(n, 1, 1)
    D_tilda = torch.diag_embed(torch.sum(A, 2).pow(-0.5))
    A_hat = D_tilda.bmm(A_tilda).bmm(D_tilda)
    return A_hat

def get_dataset(n_train=1024, n_valid=1024, n_nodes=5):
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    A = A.to(dtype)
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    data = Normalize_Adj(data)
    print(data)
    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    train_data = TensorDataset(data[0])
    valid_data = TensorDataset(data[1])
    return train_data, valid_data

def _test():
    t, v = get_dataset(2, 2)
    for d in t.tensors + v.tensors:
        print(d.size())
_test()
