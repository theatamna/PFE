import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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

class attention_layer(nn.Module):
    # THIS IS A DRAFT
    def __init__(self, in_features, out_features, alpha, dropout):
        super(attention_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        gain = torch.nn.init.calculate_gain('leaky_relu', alpha)

        self.fc = nn.Linear(in_features, out_features, bias=False) # First linear transformation
        self.a = nn.Parameter(torch.zeros((2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=gain)

        self.LeakyRelu = nn.LeakyReLU(alpha)

    def node_neighbors_features(self, batch_graphs, batch_features):
        '''
        batch_graphs: batch of adjacency matrices (tensor)
        batch_features: features of each node in every graph (tensor)

        returns: concatenated neighboring features for each node (tuple of tensors)
        '''
        # could be better (probably)
        n_features = torch.unsqueeze(batch_features, 1)
        n_features = n_features.repeat(1, batch_graphs.shape[1], 1, 1)
        degrees = batch_graphs.sum(dim=2).flatten().to(torch.int)
        temp = n_features[batch_graphs==1]
        feat = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1], -1)
        feat = torch.repeat_interleave(feat, degrees.to(torch.long), dim=0)
        concat_features = torch.cat((feat, temp), dim=1)
        return concat_features

    def forward(self, batch_graphs, batch_features):
        batch_size, n_nodes, _ = batch_graphs.shape
        # Reshape features to pass them to linear layer
        mod_features = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1], -1)

        # Linear transformation (dimension of features: in_features --> out_features)
        mod_features = self.fc(mod_features)

        # Back to the original batched shape
        mod_features = mod_features.reshape(batch_features.shape[0], batch_features.shape[1], -1)

        concat_features = self.node_neighbors_features(batch_graphs, mod_features)
        scores = self.LeakyRelu(torch.mm(concat_features, self.a))

        # Split the output: each chunk contains attention coefficients for a single node
        degrees = batch_graphs.sum(dim=2).flatten().to(torch.int)
        scores = torch.split(scores, split_size_or_sections=degrees.tolist(), dim=0)
        # scores = torch.transpose(scores, 1, 2) # Not sure if this line works, haven't tested yet
        scores = pad_sequence(scores, batch_first=True, padding_value=-9e15)
        scores = F.softmax(scores).flatten()
        scores = scores[scores>0]
        out = scores.unsqueeze(1)*concat_features[:,self.out_features:]
        out = pad_sequence(torch.split(out, degrees.tolist(), dim=0),
                                        batch_first=True,
                                        padding_value=0)
        out = torch.sum(out, dim=1)
        return out.reshape(batch_size, n_nodes, -1)

class GIN(nn.Module):
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

        # Input MLP layer
        self.mlp_layers.append(MLP(n_mlp_layers, input_dim, hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        for i in range(1, self.n_gnn_layers-1):
            self.mlp_layers.append(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Prediction layers for MLPs (hidden_dim --> output_dim and input_dim --> output_dim)
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
        inter_out = batch_features
        layer_scores = torch.empty((self.n_gnn_layers,
                                    batch_graphs.shape[0]*batch_graphs.shape[1],
                                    self.output_dim))
        for layer in range(self.n_gnn_layers-1):
            input = self.sum_neighbouring_features(batch_graphs, inter_out, layer)
            # Intermediate layers' outputs
            layer_scores[layer,:,:] = F.dropout(self.mlp_pred[layer](input), self.dropout)
            out = self.mlp_layers[layer](input)
            inter_out = F.relu(self.batch_norms[layer](out)).reshape(batch_graphs.shape[0], batch_graphs.shape[1], -1)

        # Last layer (the one without batch_norm)
        input = self.sum_neighbouring_features(batch_graphs, inter_out, self.n_gnn_layers-1)
        layer_scores[self.n_gnn_layers-1,:,:] = (F.dropout(self.mlp_pred[self.n_gnn_layers-1](input), self.dropout))

        # Readout (sum) and concat.
        layer_scores = layer_scores.reshape(self.n_gnn_layers, batch_graphs.shape[0], batch_graphs.shape[1], self.output_dim)
        node_scores = layer_scores[self.n_gnn_layers-1, :, :, :] # Added this line, returns vector of scores for each node, just like GCN
        layer_scores = torch.sum(layer_scores, 2)
        layer_scores = layer_scores.transpose(1, 2)
        layer_scores = layer_scores.reshape(-1, batch_graphs.shape[0])
        layer_scores = layer_scores.transpose(0, 1)

        return layer_scores, node_scores
