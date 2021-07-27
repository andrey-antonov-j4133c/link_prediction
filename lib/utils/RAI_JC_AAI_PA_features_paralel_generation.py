import ray
import pandas as pd
import networkx as nx


@ray.remote
def RAI(G, edge_list): return pd.DataFrame(list(nx.resource_allocation_index(G, edge_list)), columns=['node1', 'node2', 'RAI'])

@ray.remote
def JC(G, edge_list): return pd.DataFrame(list(nx.jaccard_coefficient(G, edge_list)), columns=['node1', 'node2', 'JC'])

@ray.remote
def AAI(G, edge_list): return pd.DataFrame(list(nx.adamic_adar_index(G, edge_list)), columns=['node1', 'node2', 'AAI'])

@ray.remote
def PA(G, edge_list): return pd.DataFrame(list(nx.preferential_attachment(G, edge_list)), columns=['node1', 'node2', 'PA'])


def calculate_features(G, edge_list):
    ray.init()
    ray_ids = [RAI.remote(G, edge_list), JC.remote(G, edge_list), AAI.remote(G, edge_list), PA.remote(G, edge_list)]
    results = ray.get(ray_ids)
    ray.shutdown()
    return results


