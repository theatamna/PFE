import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# Setting up the default data type
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32
torch.set_default_tensor_type(FloatTensor)

def train_GNN(model, train_loader, valid_loader, optimizer, criterion, num_epochs, device):

    def test_GNN(model, valid_loader, device):
        model = model.to(dtype).to(device=device)
        with torch.no_grad():
            correct = 0
            total = 0
            for Adj, Feat, labels in valid_loader:
                Adj = Adj.to(dtype).to(device=device)
                Feat = Feat.to(dtype).to(device=device)
                labels = labels.to(torch.long).to(device=device)
                _, outputs = model(Adj, Feat)
                _, predicted = torch.max(outputs, 2)
                total += labels.numel()
                correct += (predicted == labels).sum()

        # print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))
        return 100 * correct / total

    train_log = torch.zeros((num_epochs, 4), dtype=dtype, requires_grad=False)
    model = model.to(dtype).to(device=device)
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        for i, (Adj, Feat, labels) in enumerate(train_loader): # Need to change this later
            Adj = Adj.to(dtype).to(device=device)
            Feat = Feat.to(dtype).to(device=device)
            labels = labels.to(torch.long).to(device=device)

            # Forward pass
            _, outputs = model(Adj, Feat)
            loss = criterion(outputs.transpose(1, 2), labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).sum()
            total += labels.numel()

        train_log[epoch, 0] = epoch
        train_log[epoch, 1] = loss.item()
        train_log[epoch, 2] = (100 * correct / total)
        train_log[epoch, 3] = test_GNN(model, valid_loader, device)
        print('Epoch [{}/{}], Loss: {:.4f}, train_acc: {:.1f}, valid_acc: {:.1f}'.format(epoch + 1, num_epochs, loss, train_log[epoch, 2], train_log[epoch, 3]))

    train_log = train_log.detach().cpu().numpy()
    return train_log

def plot_learning_curves(train_log):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.tight_layout()

    ax[0].grid()
    ax[1].grid()

    ax[0].plot(train_log[:, 0], train_log[:, 1])
    ax[0].set(xlabel="epochs", ylabel="loss")
    ax[0].legend()

    ax[1].set_ylim(bottom=0, top=100)
    ax[1].plot(train_log[:, 0], train_log[:, 2], marker="v", markevery=20, label="train_acc")
    ax[1].set(xlabel="epochs", ylabel="train_acc")
    ax[1].plot(train_log[:, 0], train_log[:, 3], marker="s", markevery=20, label="valid_acc")
    ax[1].set(xlabel="epochs", ylabel="valid_acc")
    ax[1].legend()

    plt.show()
