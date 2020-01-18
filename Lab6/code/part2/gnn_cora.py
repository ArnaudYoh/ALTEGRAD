"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss
from sklearn.manifold import TSNE

from utils import load_data, accuracy, normalize_adjacency
from models import GNN

# Hyperparameters
epochs = 100
n_hidden_1 = 64
n_hidden_2 = 32
learning_rate = 0.01
dropout_rate = 0.5

# Read data
features, adj, class_labels = load_data()
n = adj.shape[0]  # Number of nodes
n_class = class_labels.shape[1]

adj = normalize_adjacency(adj)  # Normalize adjacency matrix

# Yields indices to split data into training, validation and test sets
idx = np.random.permutation(n)
idx_train = idx[:int(0.6 * n)]
idx_val = idx[int(0.6 * n):int(0.8 * n)]
idx_test = idx[int(0.8 * n):]

# Transform the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features)
y = torch.LongTensor(np.argmax(class_labels, axis=1))
adj = torch.FloatTensor(adj)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], y[idx_train])
    acc_train = accuracy(output[idx_train], y[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, _ = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], y[idx_val])
    acc_val = accuracy(output[idx_val], y[idx_val])
    print('Epoch: {:03d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output, embeddings = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], y[idx_test])
    acc_test = accuracy(output[idx_test], y[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return embeddings[idx_test]


# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()

# Testing
embeddings_test = test()

############## Task 13
# Transforms torch tensor to numpy matrix
embeddings_test = embeddings_test.detach().numpy()
##################
# your code here #
##################

# Projects the emerging representations to two dimensions using t-SNE
tsne = TSNE(2)
embeddings_test_2d = tsne.fit_transform(embeddings_test)
##################
# your code here #
##################


labels = np.argmax(class_labels[idx_test, :], axis=1)
unique_labels = np.unique(labels)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig, ax = plt.subplots()
for i in range(unique_labels.size):
    idxs = [j for j in range(labels.size) if labels[j] == unique_labels[i]]
    ax.scatter(embeddings_test_2d[idxs, 0],
               embeddings_test_2d[idxs, 1],
               c=colors[i],
               label=i,
               alpha=0.7,
               s=10)

ax.legend(scatterpoints=1)
fig.suptitle('T-SNE Visualization of the nodes of the test set', fontsize=12)
fig.set_size_inches(15, 9)
plt.savefig('Cora_tsne.png')
plt.show()

