"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from deepwalk import deepwalk
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)

############## Task 5
colors = np.array([(1., .4, .4) for _ in range(len(y))])
indexes = y == 1
indexes = np.array(indexes)
colors[indexes] = (0.4, .4, 1.)

nx.draw_networkx(G, node_color=colors)
plt.show()
##################
# your code here #
##################


############## Task 6
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8 * n)]
idx_test = idx[int(0.8 * n):]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]

############## Task 7
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy on test - DeepWalk", accuracy_score(y_test, y_pred))

y_pred = model.predict(X_train)
print("Accuracy on train - DeepWalk", accuracy_score(y_train, y_pred))
##################
# your code here #
##################


############## Task 8
model = LogisticRegression()
spectral_embedding = SpectralEmbedding()
X_train = spectral_embedding.fit_transform(X_train)
X_test = spectral_embedding.fit_transform(X_test)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy on test - Spectral", accuracy_score(y_test, y_pred))

y_pred = model.predict(X_train)
print("Accuracy on train - Spectral", accuracy_score(y_train, y_pred))

##################
# your code here #
##################
