from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class NodeMinibatchIterator(object):

    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx,
            placeholders, batch_size=100,
            max_degree=25, **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.adj, self.deg = self.construct_adj()
        self.train_nodes = set(G.nodes())
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]


    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})

        return feed_dict

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = list(self.nodes)
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        batch1 = [self.id2idx[n] for n in val_nodes]
        batch_size = len(batch1)
        return batch_size, batch1, (iter_num+1)*size >= len(node_list), val_nodes
