import ast
import logging as log
import os
from typing import Tuple

import networkx as nx
import numpy as np
import modin.pandas as pd

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

            train_1 = pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/train_1.csv')
            test_1 = pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/test_1.csv')
            test_2 = pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/test_2.csv')

            attributes = \
                pd.read_csv(PRE_COMPUTED_PATH + self.args['dataset_name'] + '/attributes.csv', converters={"attrs": ast.literal_eval}) if self.attributed else None

            return train_1, test_1, test_2, attributes

        G = nx.convert_matrix.from_numpy_matrix(self.A)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)

        train1, test1, test2 = self._sample_edges(G.edges, CG.edges, RANDOM_SEED)

        attributes = pd.DataFrame(pd.Series(self.H, name='attrs')) if self.attributed else None

        if len(TOPOLOGICAL_FEATURE_NAMES) > 0:
            log.info(f'Calculating topological features: {TOPOLOGICAL_FEATURE_NAMES}')

            train1 = self._calculate_features(train1, G)
            test1 = self._calculate_features(test1, G)
            test2 = self._calculate_features(test2, G)

            log.info('Done calculating topological features')

        return \
            train1.dropna(), \
            test1.dropna(),\
            test2.dropna(), \
            attributes.dropna() if self.attributed else None

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

        df = pd.concat([df1, df0]).reset_index(drop=True)

        # randomly choosing tr1% (70% by default) of all edges
        df_tr_1 = df.sample(n=int(len(df) * tr1), random_state=seed)

        remainder = pd.concat([df, df_tr_1]).drop_duplicates(keep=False).reset_index(drop=True)

        # balancing the edges and non-edges
        min_length = np.min([len(df_tr_1[df_tr_1['goal'] == 0]), len(df_tr_1[df_tr_1['goal'] == 1])])
        df_tr_1 = pd.concat(
            [df_tr_1[df_tr_1['goal'] == 0].sample(n=min_length, random_state=seed),
             df_tr_1[df_tr_1['goal'] == 1].sample(n=min_length, random_state=seed)]
            ).reset_index(drop=True)

        # randomly choosing ts1% (15% by default) of all edges
        df_ts_1 = remainder.sample(n=int(len(df) * ts1), random_state=seed).reset_index(drop=True)

        # the rest is test 2
        df_ts_2 = pd.concat([remainder, df_ts_1], ignore_index=True).drop_duplicates(keep=False).reset_index(drop=True)

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
