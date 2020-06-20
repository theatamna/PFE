import sys
sys.path.append("models/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

# Setting up the default data type
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float64
torch.set_default_tensor_type(FloatTensor)

class GIN(nn.Module):
    def __init__(self, n_gnn_layers, n_mlp_layers, input_dim, hidden_dim,
                        output_dim, learn_eps, dropout, attention=False):
        '''
        n_gnn_layers: number of MLPs in the GNN
        n_mlp_layers: number of layers in the MLPs (without the input layer)
        input_dim: dimension of input features of the first MLP
        hidden_dim: dimension of all hidden layers of the MLPs
        output_dim: number of classes for prediction
        learn_eps: whether epsilon is fixed beforehand or we learn epsilon by gradient descent
        dropout: dropout rate
        '''
        super().__init__()
        self.n_gnn_layers = n_gnn_layers
        self.n_mlp_layers = n_mlp_layers
        self.learn_eps = learn_eps
        self.dropout = dropout
        self.output_dim = output_dim
        self.attention = attention
        self.eps = nn.Parameter(torch.zeros(self.n_gnn_layers))

        # List of MLPs
        self.mlp_layers = torch.nn.ModuleList()

        # List of batch norm. layers applied right after each MLP (except the last one)
        self.batch_norms = torch.nn.ModuleList()

        # List of attention layers
        if self.attention:
            self.attention_layers = torch.nn.ModuleList()

        # My vision (ida rani fahem mli7): if (for example) n_gnn_layers = 5, then we need 5 attention layers, 5 MLPs, 4 batch norm. layers
        # The MLP is an integral part of the GIN embedding equation, tsema for each "GIN layer" as a whole lazem 1 MLP operation
        # (+ 1 attention operation if we're talking about GINA). Batch norm. makanch after the last layer, hence only 4
        # in the example I gave. Prediction should be done only once, hence the change to only 1 prediction layer.
        # It comes after concatenating the sums of the input features and the outputs from each of the GIN layers.
        # Ex: n_gnn_layers = 5 => 6 tensors to concatenate: 1st is the sum of input features + 5 sums of the outputs of
        # each GIN layer.
        # I made changes which I believe reflect my (newfound) understanding of GIN, but I haven't tested (it's getting late).
        # I'll leave comments fle3fayes li bedelt so you can understand my train of thought.
        # I'll suppose for the following comments that n_gnn_layers = 5 to better illustrate things.
	# We'll use hidden_dim as an output dimension for every MLP (and also, for every single hidden layer of every MLP) bach ma tetlefch and to simplify stuff

        # First MLP (input-level) + attention layer (index 0)
        self.mlp_layers.append(MLP(n_mlp_layers, input_dim, hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        if self.attention:
            self.attention_layers.append(attention_layer(input_dim))

        # Intermediate MLPs + attention layers (indices 1, 2, 3)
        for i in range(1, self.n_gnn_layers-1):
            self.mlp_layers.append(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if self.attention:
                self.attention_layers.append(attention_layer(hidden_dim))

        # Last MLP (the one not followed by batch norm.) + last attention layer (index 4 for attention and MLP layers, but batch norm. stops at 3)
        self.mlp_layers.append(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
        if self.attention:
            self.attention_layers.append(attention_layer(hidden_dim))

        # Prediction layer: only comes after the last MLP. I believe dropout should only be applied on top of this layer
        # TEST
        self.pred_layer = nn.Linear(input_dim + n_gnn_layers * hidden_dim, output_dim) # Its input dimension is the result of the concatenation as described

        # TEST

    def sum_neighbouring_features(self, batch_graphs, batch_features, layer_num):
        b_idd = torch.eye(batch_graphs.shape[1]).repeat(batch_graphs.shape[0], 1, 1)
        if self.learn_eps:
            # Adding self-loops (with epsilon)
            b_graphs = batch_graphs + b_idd * (1 + self.eps[layer_num])
            input = torch.bmm(b_graphs, batch_features)
        else:
            # Adding only self-loops
            b_graphs = batch_graphs + b_idd
            input = torch.bmm(b_graphs, batch_features)
        return input.reshape(-1, batch_features.shape[2])

    def forward(self, batch_graphs, batch_features):
        inter_out = batch_features
        layer_scores = [] # Turned this into a list, to which I'll append the sums from each GIN layer output (+ sum of input features as a first step)
        layer_scores.append(batch_features.sum(1))
        for layer in range(self.n_gnn_layers-1): # Tsema for indices 0, 1, 2, 3
            if self.attention:
                att_scores = self.attention_layers[layer](batch_graphs, inter_out)
                adj_mats = batch_graphs * att_scores
            else:
                adj_mats = batch_graphs
            input = self.sum_neighbouring_features(adj_mats, inter_out, layer)

            out = self.mlp_layers[layer](input)
            inter_out = F.relu(self.batch_norms[layer](out)).reshape(batch_graphs.shape[0], batch_graphs.shape[1], -1)
            layer_scores.append(inter_out.sum(1))

        # Last MLP (the one without batch norm.) (index 4 for attention and MLP, no batch norm.)
        if self.attention:
            att_scores = self.attention_layers[self.n_gnn_layers-1](batch_graphs, inter_out)
            adj_mats = batch_graphs * att_scores
        else:
            adj_mats = batch_graphs
        input = self.sum_neighbouring_features(adj_mats, inter_out, self.n_gnn_layers-1)

        out = self.mlp_layers[self.n_gnn_layers-1](input)
        inter_out = F.relu(out).reshape(batch_graphs.shape[0], batch_graphs.shape[1], -1)
        layer_scores.append(inter_out.sum(1))

        # Now, layer_scores should be a list of 6 2D tensors,
        # with each line (from each tensor) being a graph-level readout (from the corresponding layer)
        # Then, we concatenate along dim=1, then pass the resulting tensor to a linear layer (+ dropout) for classification

        # Concatenation
        graph_readouts = torch.cat(layer_scores, 1) # Not sure if this is how to concatenate a list of tensors

        _ = 0
        # Prediction layer
    
        graph_scores = F.dropout(self.pred_layer(graph_readouts), self.dropout)  

        return _, graph_scores # the "_" only bach ma nbedlouch the training code and immediatly test whatever we change here
        