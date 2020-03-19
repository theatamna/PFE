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
        super().__init__()
        self.n_gnn_layers = n_gnn_layers
        self.n_mlp_layers = n_mlp_layers
        self.learn_eps = learn_eps
        self.dropout = dropout
        self.output_dim = output_dim
        self.eps = nn.Parameter(torch.zeros(self.n_gnn_layers))


        # List of MLPs
        self.mlp_layers = torch.nn.ModuleList()

        # Batchnorms applied to the final layer of each MLP
        self.batch_norms = torch.nn.ModuleList()

        # input MLP layer
        self.mlp_layers.append(MLP(n_mlp_layers, input_dim, hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
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
            b_graphs = batch_graphs + b_idd * (1 + self.eps[layer_num])
            input = torch.bmm(b_graphs, batch_features)
        else:
            # Adding only self Loops
            b_graphs = batch_graphs + b_idd
            input = torch.bmm(b_graphs, batch_features)
        return input.reshape(-1, batch_features.shape[2])

    def forward(self, batch_graphs, batch_features):
        # This is a DRAFT of the forward function
        inter_out = batch_features
        layer_scores = torch.empty((self.n_gnn_layers,
                                    batch_graphs.shape[0]*batch_graphs.shape[1],
                                    self.output_dim))
        for layer in range(self.n_gnn_layers-1):
            input = self.sum_neighbouring_features(batch_graphs, inter_out, layer)
            # intermediate layers' outputs
            layer_scores[layer,:,:] = F.dropout(self.mlp_pred[layer](input), self.dropout)
            #
            out = self.mlp_layers[layer](input)
            inter_out = F.relu(self.batch_norms[layer](out)).reshape(batch_graphs.shape[0], batch_graphs.shape[1], -1)
        # last layer (the one without batch_norm)
        input = self.sum_neighbouring_features(batch_graphs, inter_out, self.n_gnn_layers-1)
        layer_scores[self.n_gnn_layers-1,:,:] = (F.dropout(self.mlp_pred[self.n_gnn_layers-1](input), self.dropout))
        out = self.mlp_layers[-1](input)
        inter_out = F.relu(out).reshape(batch_graphs.shape[0], batch_graphs.shape[1], -1)

        # Readout (sum) and concat.
        layer_scores = layer_scores.reshape(self.n_gnn_layers, batch_graphs.shape[0], batch_graphs.shape[1], self.output_dim)
        layer_scores = torch.sum(layer_scores, 2)
        layer_scores = layer_scores.transpose(1, 2)
        layer_scores = layer_scores.reshape(-1, batch_graphs.shape[0])
        layer_scores = layer_scores.transpose(0, 1)

        return layer_scores


model = GIN(5, 6, 3, 10, 4, True, 0.5)
A = torch.randint(2, [10, 5, 5])
A = A.to(dtype)
upper_tr = torch.triu(A, diagonal=1)
A =  upper_tr + torch.transpose(upper_tr, 1, 2)
B = torch.randint(5, [10, 5, 3], dtype=dtype)
print(model.forward(A,B).shape)
