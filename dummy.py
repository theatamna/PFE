import torch
from torch.utils.data import TensorDataset

def get_dataset(n_train=1024, n_valid=1024, n_nodes=5):
    A = torch.randint(2, [n_train + n_valid, n_nodes, n_nodes])
    upper_tr = torch.triu(A, diagonal=1)
    data =  upper_tr + torch.transpose(upper_tr, 1, 2)
    data = torch.split(data, split_size_or_sections=[n_train, n_valid], dim=0)
    train_data = TensorDataset(data[0])
    valid_data = TensorDataset(data[1])
    return train_data, valid_data

def _test():
    t, v = get_dataset()
    for d in t.tensors + v.tensors:
        print(d.size())
_test()
