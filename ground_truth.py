import networkx as nx
import random


def LFR(n=500, t1=3, t2=1.5, m=0.1):
    seed = random.randint(1, 10)
    G = nx.LFR_benchmark_graph(n, t1, t2, m, average_degree=5, min_community=20, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, nx.complement(G)
