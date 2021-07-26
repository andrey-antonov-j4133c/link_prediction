import networkx as nx
import pandas as pd
import numpy as np


class LFR:
    def __init__(self, n=500, t1=3, t2=1.5, m=0.1, seed=0) -> None:
        self.feature_names = ['RAI', 'JC', 'AAI', 'PA']

        G = nx.LFR_benchmark_graph(n, t1, t2, m, average_degree=5, min_community=20, seed=seed)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)

        self.sample_edges_df(G, CG)
        self.calc_features(G)

        
    def sample_edges_df(self, G, CG) -> None:
        # df from graph
        df = pd.DataFrame(list(G.edges()), columns=['node1', 'node2'])
        df[['goal']] = 1
        comp_df = pd.DataFrame(list(CG.edges()), columns=['node1', 'node2'])
        comp_df[['goal']] = 0
        df = df.append(comp_df)

        # split node pairs with edge existing (1) and non-existing (0)
        df0 = df[df['goal'] == 0]
        df1 = df[df['goal'] == 1]

        # number of sampled edges/non-eges
        n = np.min([len(df0), len(df1)])  

        # sampling n edges and non-edges
        self.df = pd.concat([
            df0.sample(int(n/2)), 
            df1.sample(int(n/2))
        ])


    def calc_features(self, G) -> None:
        self.features_df = self.df[['node1', 'node2']]

        self.features_df['RAI'] = self.features_df.apply(
            lambda row: list(nx.resource_allocation_index(G, [(row['node1'], row['node2'])]))[0][2],
            axis=1
        )

        self.features_df['JC'] = self.features_df.apply(
            lambda row: list(nx.jaccard_coefficient(G, [(row['node1'], row['node2'])]))[0][2],
            axis=1
        )

        self.features_df['AAI'] = self.features_df.apply(
            lambda row: list(nx.adamic_adar_index(G, [(row['node1'], row['node2'])]))[0][2],
            axis=1
        )

        self.features_df['PA'] = self.features_df.apply(
            lambda row: list(nx.preferential_attachment(G, [(row['node1'], row['node2'])]))[0][2],
            axis=1
        )

        for column_name in ['PA']:
            self.features_df[column_name] -= self.features_df[column_name].min()
            self.features_df[column_name] /= self.features_df[column_name].max()

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_features_df(self) -> pd.DataFrame:
        return self.features_df