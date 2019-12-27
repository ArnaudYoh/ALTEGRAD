"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    walk = [node]

    for _ in range(walk_length):
        children = list(G.neighbors(node))
        next_node = children[randint(0, len(children) - 1)]
        walk.append(next_node)
        node = next_node
    ##################
    # your code here #
    ##################

    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    for i, node in enumerate(list(G.nodes())):
        for _ in range(num_walks):
            walks.append(random_walk(G, node, walk_length))
        if (i+1) % 1000 == 0:
            print("Done with", i+1, "nodes")
    ##################
    # your code here #
    ##################

    return walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
