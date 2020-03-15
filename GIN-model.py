import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float

class MLP(nn.Module):
    def __init__(self, dims):
        '''
        dims: a tuple containing the dimension of each layer
            (input_dim, hidden_dim1, ... , output_dim)
        '''
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1

        self.linear_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(1, self.n_layers + 1):
            self.linear_layers.append(nn.Linear(dims[i-1], dims[i]))

        for i in range(1, self.n_layers):
            self.batch_norms.append(nn.BatchNorm1d((dims[i])))

    def forward(self, x):
        out = x
        for i in range(self.n_layers - 1):
            out = F.relu(self.batch_norms[i](self.linear_layers[i](out)))
        return self.linear_layers[self.n_layers - 1](out)

# dims = (5, 10, 20, 30, 40, 50, 5)
# model = MLP(dims)
# print(model)
# A = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]], dtype=dtype)
# print(model.forward(A))
