"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

############## Task 1

##################
# your code here #
##################
print("TASK 1")
G = nx.read_edgelist(path='CA-HepTh.txt', delimiter='\t')
print("Number of Nodes", G.number_of_nodes())
print("Number of Edges", G.number_of_edges())
print()

############## Task 2

##################
# your code here #
##################
print("TASK 2")
print("Number of Connected Components", nx.number_connected_components(G))
largest_component = G.subgraph(max(nx.connected_components(G), key=len))
print("Ratio of Edges {:.2f}".format(largest_component.number_of_edges() / G.number_of_edges()))
print("Ratio of Nodes {:.2f}".format(largest_component.number_of_nodes() / G.number_of_nodes()))
print()

############## Task 3
# Degree
print("TASK 3")
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################
print("Mean degree of the graph (rounded) {:.2f}".format(np.mean(degree_sequence)))
print("Max degree of the graph", np.max(degree_sequence))
print("Min degree of the graph", np.min(degree_sequence))

print()

############## Task 4

##################
# your code here #
##################

values = nx.degree_histogram(G)

plt.figure(figsize=(15, 6))
plt.plot(values)
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.title("Distribution of Node Degrees")
plt.show()
