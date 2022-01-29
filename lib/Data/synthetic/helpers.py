import os
import networkx as nx


def write_features(X, path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'Features.csv', "w") as f:
        for row in X:
            row_str = ','.join([str(itm) for itm in row])
            f.write(f'{row_str}\n')


def write_network(G, path):
    if not os.path.exists(path):
        os.makedirs(path)
    nx.write_adjlist(G, path + 'G.csv')