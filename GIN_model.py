import torch
import torch.nn as nn
import torch.nn.functional as F
from dummy import * # for testing purposes

dtype = torch.float

class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        '''
        n_layers: number of layers in the MLP (without the input layer)
        input_dim: dimension of input features
        hidden_dim: dimension of all hidden layers
        output_dim: number of classes for prediction
        '''
        super(MLP, self).__init__()
        self.n_layers = n_layers

        self.linear_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(1, self.n_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers.append(nn.Linear(hidden_dim, output_dim))

        for i in range(self.n_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        out = x
        for i in range(self.n_layers - 1):
            out = F.relu(self.batch_norms[i](self.linear_layers[i](out)))
        return self.linear_layers[self.n_layers - 1](out)

# model = MLP(5, 5, 10, 4)
# print(model)
# A = torch.eye(5).repeat(10, 1, 1)
# print(A.shape)
# print(model.forward(A))

class GIN(nn.Module):
    # Still needs some work
    def __init__(self, n_gnn_layers, n_mlp_layers, input_dim, hidden_dim, output_dim, learn_eps, dropout):
        '''
        n_gnn_layers: number of MLPs in the GNN
        n_mlp_layers: number of layers in the MLP (without the input layer)
        input_dim: dimension of input features of the first MLP
        hidden_dim: dimension of all hidden layers of the MLPs
        output_dim: number of classes for prediction of the last MLP
        learn_eps: whether epsilon is fixed beforehand or we learn epsilon by gradient descent
        dropout: dropout rate
        '''
        super(GIN, self).__init__()
        self.n_gnn_layers = n_gnn_layers
        self.n_mlp_layers = n_mlp_layers
        self.learn_eps = learn_eps
        self.dropout = dropout
        self.eps = nn.Parameter(torch.zeros(self.n_gnn_layers - 1))


        # List of MLPs
        self.mlp_layers = torch.nn.ModuleList()

        # Batchnorms applied to the final layer of each MLP
        self.batch_norms = torch.nn.ModuleList()

        # input MLP layer
        self.mlp_layers.append(MLP(n_mlp_layers, input_dim, hidden_dim, hidden_dim))
        for i in range(1, self.n_gnn_layers-1):
            self.mlp_layers.append(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # prediction layers for MLPs (hidden_dim --> output_dim and input_dim --> output_dim)
        self.mlp_pred = torch.nn.ModuleList()
        self.mlp_pred.append(nn.Linear(input_dim, output_dim))
        for i in range(1, n_gnn_layers):
            self.mlp_pred.append(nn.Linear(hidden_dim, output_dim))

    def sum_neighbouring_features(self, batch_graphs, batch_features, layer_num):
        b_idd = torch.eye(batch_graphs.shape[1]).repeat(batch_graphs.shape[0], 1, 1)
        if self.learn_eps:
            # Adding self loops (with epsilon)
            b_graphs = batch_graphs + batch_idd * (1 + self.eps[layer])
            input = torch.bmm(b_graphs, batch_features)
        else:
            # Adding only self Loops
            b_graphs = batch_graphs + batch_idd
            input = torch.bmm(b_graphs, batch_features)

        return output

    def forward(self, batch_features, batch_graphs):
        # This is a DRAFT of the forward function
        '''
        '''
        # batch_idd = torch.eye(batch_graphs.shape[1]).repeat(batch_graphs.shape[0], 1, 1)
        # for layer in range(self.n_gnn_layers):
        #     if self.learn_eps:
        #         out = self.mlp_layers[layer](batch_idd*(1+self.eps[layer])*batch_features)
        #         out = F.relu(self.batch_norms[layer](out))
        return out
# tests
# model = GIN(5, 6, 5, 10, 4, True, 0.5)
# print(model)
# A = torch.randint(2, [10, 5, 5])
# A = A.to(dtype)
# upper_tr = torch.triu(A, diagonal=1)
# A =  upper_tr + torch.transpose(upper_tr, 1, 2)
# B = torch.randint(5, [10, 5, 3], dtype=dtype)
