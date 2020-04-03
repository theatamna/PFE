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
dtype = torch.float32
torch.set_default_tensor_type(FloatTensor)

def Normalize_Adj(A):
    A_tilda = A + torch.eye(A.shape[1]).repeat(A.shape[0], 1, 1)
    D_tilda = torch.diag_embed(torch.sum(A_tilda, 2).pow(-0.5))
    A_hat = D_tilda.bmm(A_tilda).bmm(D_tilda)
    return A_hat

class TwoLayerGCN(nn.Module):
  def __init__(self, input_dim=5, hidden_dim=10, n_classes=4, dropout=0.5):
    super().__init__()
    self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
    self.gc2 = GraphConvolutionLayer(hidden_dim, n_classes)
    self.dropout = dropout

  def forward(self, X, A):
    out = self.gc1(X, Normalize_Adj(A))
    out = F.relu(out)
    out = F.dropout(out, self.dropout)
    out = self.gc2(out, Normalize_Adj(A))
    return out