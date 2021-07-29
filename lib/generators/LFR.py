from os import terminal_size

from pandas.core.indexes.numeric import IntegerIndex
import networkx as nx
import pandas as pd
import numpy as np
import random

from lib.generators.generator_base import Generator


class LFR(Generator):
    def __init__(self, feature_names=['RAI', 'JC', 'AAI', 'PA'], features_to_normalize=['PA'], n=500, t1=3, t2=1.5, m=0.1, seed=0) -> None:
        self.feature_names = feature_names
        print('Generating LFR graph...')
        G = nx.LFR_benchmark_graph(n, t1, t2, m, average_degree=3, min_community=10, seed=seed)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)
        print('Graph generated!')

        print('Train/test splitting...')
        TRAIN1, TEST1, TEST2 = self.sample_edges(G.edges, CG.edges, seed)
        print('Done splitting the data!')

        print(f'Calculating {feature_names} features...')
        self.TRAIN1_DF = self.calculate_features(TRAIN1, G, feature_names, features_to_normalize)
        self.TEST1_DF = self.calculate_features(TEST1, G, feature_names, features_to_normalize)
        self.TEST2_DF = self.calculate_features(TEST2, G, feature_names, features_to_normalize)
        print('Done calculating')

    def sample_edges(self, G_edges, CG_edges, seed, tr1=0.7, ts1=0.15):
        df1 = pd.DataFrame()
        df0 = pd.DataFrame()

        df1['node1'] = pd.Series([e[0] for e in G_edges])
        df1['node2'] = pd.Series([e[1] for e in G_edges])
        df1['goal'] = pd.Series(np.ones(len(G_edges)))

        df0['node1'] = pd.Series([e[0] for e in CG_edges])
        df0['node2'] = pd.Series([e[1] for e in CG_edges])
        df0['goal'] = pd.Series(np.zeros(len(CG_edges)))

        df = pd.concat([df1, df0], ignore_index=True)
        
        # randomly choosing tr1% (70% by default) of all edges
        df_tr_1 = df.sample(n=int(len(df)*tr1), random_state=seed)

        remainder = pd.concat([df, df_tr_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        # balancing the edges and non-edges
        min_length = np.min([len(df_tr_1[df_tr_1['goal'] == 0]), len(df_tr_1[df_tr_1['goal'] == 1])])
        df_tr_1 = pd.concat([\
            df_tr_1[df_tr_1['goal'] == 0].sample(n=min_length, random_state=seed),\
            df_tr_1[df_tr_1['goal'] == 1].sample(n=min_length, random_state=seed)],\
            ignore_index=True)

        # randomly choosing ts1% (15% by default) of all edges
        df_ts_1 = remainder.sample(n=int(len(df)*ts1), random_state=seed, ignore_index=True)

        # the rest is test 2
        df_ts_2 = pd.concat([remainder, df_ts_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        return df_tr_1, df_ts_1, df_ts_2

    def calculate_features(self, df, G, feature_names, features_to_normalize):
        
        functions = {
            'RAI': nx.resource_allocation_index,
            'JC': nx.jaccard_coefficient,
            'AAI': nx.adamic_adar_index,
            'PA': nx.preferential_attachment
        }

        for feature in feature_names:
            frm = df['node1'].values
            to = df['node2'].values
            edges = [(frm[i], to[i]) for i in range(len(frm))]
            features = [f[2] for f in functions[feature](G, edges)]
            
            df[feature] = pd.Series(features)
            if feature in features_to_normalize:
                df[feature] -= df[feature].min()
                df[feature] /= df[feature].max()

        return df