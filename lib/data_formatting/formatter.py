import logging as log
import os

import networkx as nx
import numpy as np
import pandas as pd

from settings import *


class Formatter:
    def __init__(self, args: dict, attributed=True) -> None:
        self.A = None
        self.H = None
        self.y = None

        self.args = args
        self.attributed = attributed

    def load_data(self):
        if self.args['dataset_name'] in os.listdir(PRE_COMPUTED_PATH):
            log.info('Found data from pre-computed')
            log.info('Loading data now')

            return {
                'train_1': pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/train_1.csv'),
                'test_1': pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/test_1.csv'),
                'test_2': pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/test_2.csv')
            }

        G = nx.convert_matrix.from_numpy_matrix(self.A)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)

        TRAIN1, TEST1, TEST2 = self._sample_edges(G.edges, CG.edges, RANDOM_SEED)

        if self.attributed:
            TRAIN1['node_1_attrs'] = TRAIN1['node1'].apply(lambda x: self.H[x])
            TRAIN1['node_2_attrs'] = TRAIN1['node2'].apply(lambda x: self.H[x])

            TEST1['node_1_attrs'] = TEST1['node1'].apply(lambda x: self.H[x])
            TEST1['node_2_attrs'] = TEST1['node2'].apply(lambda x: self.H[x])

            TEST2['node_1_attrs'] = TEST2['node1'].apply(lambda x: self.H[x])
            TEST2['node_2_attrs'] = TEST2['node2'].apply(lambda x: self.H[x])

        if len(TOPOLOGICAL_FEATURE_NAMES) > 0:
            log.info(f'Calculating topological features: {TOPOLOGICAL_FEATURE_NAMES}')

            TRAIN1 = self._calculate_features(TRAIN1, G)
            TEST1 = self._calculate_features(TEST1, G)
            TEST2 = self._calculate_features(TEST2, G)

            log.info('Done calculating topological features')

        return {
            'train_1': TRAIN1,
            'test_1': TEST1,
            'test_2': TEST2
        }

    def _sample_edges(self, G_edges, CG_edges, seed, tr1=0.7, ts1=0.15):
        log.info('Train/test splitting the data...')

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
        df_tr_1 = df.sample(n=int(len(df) * tr1), random_state=seed)

        remainder = pd.concat([df, df_tr_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        # balancing the edges and non-edges
        min_length = np.min([len(df_tr_1[df_tr_1['goal'] == 0]), len(df_tr_1[df_tr_1['goal'] == 1])])
        df_tr_1 = pd.concat(
            [df_tr_1[df_tr_1['goal'] == 0].sample(n=min_length, random_state=seed),
             df_tr_1[df_tr_1['goal'] == 1].sample(n=min_length, random_state=seed)],
            ignore_index=True)

        # randomly choosing ts1% (15% by default) of all edges
        df_ts_1 = remainder.sample(n=int(len(df) * ts1), random_state=seed, ignore_index=True)

        # the rest is test 2
        df_ts_2 = pd.concat([remainder, df_ts_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        log.info('Success!')
        return df_tr_1, df_ts_1, df_ts_2

    def _calculate_features(self, df, G):
        functions = {
            'RAI': nx.resource_allocation_index,
            'JC': nx.jaccard_coefficient,
            'AAI': nx.adamic_adar_index,
            'PA': nx.preferential_attachment
        }

        for feature, function in functions.items():
            frm = df['node1'].values
            to = df['node2'].values
            edges = [(frm[i], to[i]) for i in range(len(frm))]
            features = [f[2] for f in function(G, edges)]

            df[feature] = pd.Series(features)
            if feature in TOPOLOGICAL_FEATURES_TO_NORMALIZE:
                df[feature] -= df[feature].min()
                df[feature] /= df[feature].max()

        return df

    def _read_data(self, args: dict):
        pass
