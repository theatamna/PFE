import torch
import torch.nn as nn
import torch.nn.functional as F

def Normalize_Adj(A):
    A_tilda = A + torch.eye(A.shape[1]).repeat(A.shape[0], 1, 1)
    D_tilda = torch.diag_embed(torch.sum(A_tilda, 2).pow(-0.5))
    A_hat = D_tilda.bmm(A_tilda).bmm(D_tilda)
    return A_hat

class GraphConvolutionLayer(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim, bias=False)
    torch.nn.init.xavier_uniform_(self.fc.weight)
  def forward(self, X, A):
    out = self.fc(X)
    out = torch.bmm(A, out)  
    return out

class TwoLayerGCN(nn.Module):
  def __init__(self, input_dim=5, hidden_dim=10, n_classes=4, dropout=0.5):
    super().__init__()
    self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
    self.gc2 = GraphConvolutionLayer(hidden_dim, n_classes)
    self.dropout = dropout

  def forward(self, X, A):
    X = X.repeat(A.shape[0], 1, 1)
    #print('X: ', X.shape)
    out = self.gc1(X, Normalize_Adj(A))
    #print('Output after 1st GCN layer: ', out.shape)
    out = F.relu(out)
    out = F.dropout(out, self.dropout)
    out = self.gc2(out, Normalize_Adj(A))
    #print('Output after 2nd GCN layer: ', out.shape)
    #out = F.softmax(out) 
    return out