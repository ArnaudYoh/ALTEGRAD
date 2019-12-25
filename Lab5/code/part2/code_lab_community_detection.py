"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

# Load the graph into an undirected NetworkX graph
G = nx.read_edgelist("./CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())

# Get giant connected component (GCC)
GCC = G.subgraph(max(nx.connected_components(G), key=len))

k = 50


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
##################
# your code here #
##################

def spectral_clustering(G, k):
    L = nx.laplacian_matrix(G).astype('float')
    eig_val, eig_vec = eigs(L)

    eig_val = eig_val.real  # Keep the real part
    eig_vec = eig_vec.real  # Keep the real part

    idx = eig_val.argsort()  # Get indices of sorted eigenvalues
    eig_vec = eig_vec[:, idx]  # Sort eigenvectors according to eigenvalues

    k_m = KMeans(n_clusters=k)
    k_m.fit(eig_vec)

    clusters = k_m.predict(eig_vec)
    clustering = {n: c for n, c in zip(G.nodes(), clusters)}

    return clustering


############## Task 6
clustering = spectral_clustering(GCC, k)


##################
# your code here #
##################


############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    n_clusters = len(set(clustering.values()))
    modularity = 0  # Initialize total modularity value
    m = G.number_of_edges()
    # Iterate over all clusters
    for i in range(n_clusters):
        node_list = [n for n, v in clustering.items() if v == i]  # Get the nodes that belong to the i-th cluster
        subG = G.subgraph(node_list)  # get subgraph that corresponds to current cluster

        # Compute contribution of current cluster to modularity as in equation 1

        ##################
        # your code here #
        lc = subG.number_of_edges()
        dc = sum([v for k, v in subG.degree])

        modularity += lc / m - (dc / (2 * m)) ** 2
        ##################

    return modularity


############## Task 8
print("Modularity Spectral Clustering: {:.3f}".format(modularity(GCC, clustering)))

# Implement random clustering
r_clustering = dict()
for node in GCC.nodes:
    r_clustering[node] = np.random.randint(0, k)
print("Modularity Random Clustering: {:.3f}".format(modularity(GCC, r_clustering)))
