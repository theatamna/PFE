import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

dtype = torch.float32
torch.set_default_tensor_type(torch.FloatTensor)

class GraphConvolutionLayer(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim, bias=False)
    torch.nn.init.xavier_uniform_(self.fc.weight)
  def forward(self, X, A):
    out = self.fc(X.reshape(-1, X.shape[2]))
    out = out.reshape(A.shape[0], A.shape[1], -1)
    out = torch.bmm(A, out)
    return out

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

    def __init__(self, in_features, alpha=0.2, dropout=0):
        super(attention_layer, self).__init__()
        self.in_features = in_features
        self.alpha = alpha
        self.dropout = dropout

        gain = torch.nn.init.calculate_gain('leaky_relu', alpha)

        self.a = nn.Parameter(torch.zeros((2*in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=gain)

        self.LeakyRelu = nn.LeakyReLU(alpha)

    def node_neighbors_features(self, batch_graphs, batch_features):
        '''
        batch_graphs: batch of adjacency matrices (tensor)
        batch_features: features of each node in every graph (tensor)

        returns: concatenated neighboring features for each node (tuple of tensors)
        '''

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
        attention_scores = torch.zeros_like(batch_graphs)

        concat_features = self.node_neighbors_features(batch_graphs, batch_features)
        scores = self.LeakyRelu(torch.mm(concat_features, self.a))

        # Split the output: each chunk contains attention coefficients for a single node
        degrees = batch_graphs.sum(dim=2).flatten().to(torch.int)
        scores = torch.split(scores, split_size_or_sections=degrees.tolist(), dim=0)
        scores = pad_sequence(scores, batch_first=True, padding_value=-9e15)
        scores = F.softmax(scores).flatten()
        scores = scores[scores>0]
        attention_scores.masked_scatter_(batch_graphs.to(torch.bool), scores)

        return attention_scores
