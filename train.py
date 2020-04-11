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

def train_GNN(model, folded_train_data, folded_valid_data, optimizer, criterion, num_epochs, device):
    n_folds = len(folded_train_data)
    def test_GNN(model, valid_loader, device):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for Adj, Feat, labels in valid_loader:
                Adj = Adj.to(dtype).to(device=device)
                Feat = Feat.to(dtype).to(device=device)
                labels = labels.to(torch.long).to(device=device)
                _, outputs = model(Adj, Feat)
                _, predicted = torch.max(outputs, 1)
                total += labels.numel()
                correct += (predicted == labels).sum()
        return 100 * correct / total

    model = model.to(dtype).to(device=device)
    train_acc_history = []
    valid_acc_history = []
    Wsave = model.get_weights()
    opt_save = optimizer.state_dict()
    for fold in range(n_folds):
        model.set_weights(Wsave)
        optimizer.load_state_dict(opt_save)
        train_log = torch.zeros((num_epochs, 4), dtype=dtype, requires_grad=False)
        for epoch in range(num_epochs):
            model.train()
            correct = 0
            total = 0
            for i, (Adj, Feat, labels) in enumerate(folded_train_data[fold]):
                Adj = Adj.to(dtype).to(device=device)
                Feat = Feat.to(dtype).to(device=device)
                labels = labels.to(torch.long).to(device=device)

                # Forward pass
                _, outputs = model(Adj, Feat)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                total += labels.numel()

            train_log[epoch, 0] = epoch
            train_log[epoch, 1] = loss.item()
            train_log[epoch, 2] = (100 * correct / total)
            train_log[epoch, 3] = test_GNN(model, folded_valid_data[fold], device)
            print('Fold no. {}, epoch [{}/{}], Loss: {:.4f}, train_acc: {:.1f}, valid_acc: {:.1f}'.format(fold + 1, epoch + 1, num_epochs, loss, train_log[epoch, 2], train_log[epoch, 3]))
        train_acc_history.append(train_log[epoch, 2])
        valid_acc_history.append(train_log[epoch, 3])
        train_log = train_log.detach().cpu().numpy()
        plot_learning_curves(train_log)

    print('Average training accuracy across the {} folds: {:.1f}'.format(n_folds, sum(train_acc_history)/len(train_acc_history)))
    print('Average validation accuracy across the {} folds: {:.1f}'.format(n_folds, sum(valid_acc_history)/len(valid_acc_history)))
