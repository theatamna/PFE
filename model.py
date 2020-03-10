import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float

class TwoLayerGCN(nn.Module):
    '''
    '''
    def __init__(self, Dim):
        super().__init__()
        self.fc1 = nn.Linear(Dim[0][0], Dim[0][1], bias=False)
        self.fc2 = nn.Linear(Dim[1][0], Dim[1][1], bias=False)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, A, X):
        out = self.fc1(torch.mm(A,X))
        out = F.relu(out)
        out = self.fc2(torch.mm(A,out))
        out = F.softmax(out) 
        return out

A = torch.tensor([[1,0,1,0],[0,1,1,0],[1,1,1,0],[0,0,0,1]], dtype=dtype)
X = torch.eye(4, dtype=dtype)
Dim = [[4,10],[10,5]]

model = TwoLayerGCN(Dim)

print(model.forward(A,X))
