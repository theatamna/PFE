import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
from sklearn.model_selection import KFold
import numpy as np

# Setting up the default data type
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float64
torch.set_default_tensor_type(FloatTensor)

def train_GNN(model, dataset, optimizer, criterion, num_epochs, batch_size, device, 
             n_folds=10, start_fold=1, save_name='_'):
    def test_GNN(model, test_loader, device):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for Adj, Feat, labels in test_loader:
                Adj = Adj.to(dtype).to(device=device)
                Feat = Feat.to(dtype).to(device=device)
                labels = labels.to(torch.long).to(device=device)
                _, outputs = model(Adj, Feat)
                _, predicted = torch.max(outputs, 1)
                total += labels.numel()
                correct += (predicted == labels).sum()
        return 100 * correct / total
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=300)
    model = model.to(dtype).to(device=device)
    train_history = []
    test_acc_history = []
    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())
    for (j, (train_index, test_index)) in list(enumerate(kf.split(dataset)))[start_fold-1:]:
        # Splitting train and test data
        train = Subset(dataset, train_index)
        test = Subset(dataset, test_index)
        trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test, batch_size=batch_size, shuffle=True)
        # Reset model (and optimizer) parameters
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)
        train_log = torch.zeros((num_epochs, 3), dtype=dtype, requires_grad=False)
        for epoch in range(num_epochs):
            model.train()
            correct = 0
            total = 0
            for i, (Adj, Feat, labels) in enumerate(trainloader):
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
            if (epoch % 10) == 0:
                print('Fold no. {}, epoch [{}/{}], Loss: {:.4f}, train_acc: {:.2f}'.format(j + 1, epoch, num_epochs, loss, train_log[epoch, 2]))
        train_log = train_log.detach().cpu().numpy()
        with open('./logs/train_log_{}_split.txt'.format(save_name), "ab") as f:
            np.savetxt(f, X=train_log, fmt="%d, %1.6e, %1.6e")
            f.write(b"\n")
            f.close()
        train_history.append(train_log)
        test_acc = test_GNN(model, testloader, device).cpu().numpy()
        test_acc_history.append(test_acc)
        with open('./logs/test_log_{}_split.txt'.format(save_name), "ab") as f:
            np.savetxt(f, X=np.array([j, test_acc]).reshape(1, 2), fmt="%d, %1.4e")
            f.close()

    for i, train_log in enumerate(train_history):
        fig, ax = plot_learning_curves(train_log)
        fig.suptitle("Learning Curves Fold no. {}".format(i+1))
        plt.show()

    print('Test accuracy for each fold:')
    print(*test_acc_history)
    print('Average test accuracy across the {} folds: {:.2f}'.format(n_folds, sum(test_acc_history)/len(test_acc_history)))
    print('Max test accuracy across the {} folds: {:.2f}'.format(n_folds, max(test_acc_history)))


def plot_learning_curves(train_log):
    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    #fig.tight_layout()
    ax[0].grid()
    ax[1].grid()

    ax[0].plot(train_log[:, 0], train_log[:, 1])
    ax[0].set(xlabel="epochs", ylabel="loss")

    ax[1].set_ylim(bottom=0, top=100)
    ax[1].plot(train_log[:, 0], train_log[:, 2])
    ax[1].set(xlabel="epochs", ylabel="train_acc")
    #ax[1].legend()

    return fig, ax
