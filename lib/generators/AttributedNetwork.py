import networkx as nx

import pandas as pd
import numpy as np
import random

from scipy.sparse import csr_matrix

from pandas.core.indexes.numeric import IntegerIndex
from os import terminal_size
from lib.generators.generator_base import Generator

import logging as log
from tqdm import tqdm


class AttributedNetwork(Generator):
    def __init__(
            self, 
            path,
            dataset_name,
            t_feature_names=['RAI', 'JC', 'AAI', 'PA'], 
            with_attributes = True,
            features_to_normalize=['PA'], 
            seed=0,
            vebrose=True) -> None:

        if vebrose:
            log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
            log.info("Verbose output.")
        else:
            log.basicConfig(format="%(levelname)s: %(message)s")

        self.t_feature_names = t_feature_names
        self.with_attributes = with_attributes

        A, H, y = self.read_data(path + dataset_name)

        G = nx.convert_matrix.from_numpy_matrix(A)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)

        TRAIN1, TEST1, TEST2 = self.sample_edges(G.edges, CG.edges, seed)

        if self.with_attributes:
            TRAIN1['node_1_attrs'] = TRAIN1['node1'].apply(lambda x: H[x])
            TRAIN1['node_2_attrs'] = TRAIN1['node2'].apply(lambda x: H[x])

            TEST1['node_1_attrs'] = TEST1['node1'].apply(lambda x: H[x])
            TEST1['node_2_attrs'] = TEST1['node2'].apply(lambda x: H[x])

            TEST2['node_1_attrs'] = TEST2['node1'].apply(lambda x: H[x])
            TEST2['node_2_attrs'] = TEST2['node2'].apply(lambda x: H[x])

        self.TRAIN1_DF = self.calculate_features(TRAIN1, G, t_feature_names, features_to_normalize)
        self.TEST1_DF = self.calculate_features(TEST1, G, t_feature_names, features_to_normalize)
        self.TEST2_DF = self.calculate_features(TEST2, G, t_feature_names, features_to_normalize)



    def read_data(self, path):
        log.info('Reading data ...')
        if not path.endswith('.npz'):
            path += '.npz'
        with np.load(path, allow_pickle=True) as loader:
            loader = dict(loader)
            A = csr_matrix((loader['adj_data'], loader['adj_indices'],
                            loader['adj_indptr']), shape=loader['adj_shape'])

            H = csr_matrix((loader['attr_data'], loader['attr_indices'],
                            loader['attr_indptr']), shape=loader['attr_shape'])

            y = loader.get('labels')

            log.info('Success!')
            return A.toarray(), [tuple(i) for i in H.toarray()], y
        


    def sample_edges(self, G_edges, CG_edges, seed, tr1=0.7, ts1=0.15):

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

        OLD = True

        if OLD:
            # randomly choosing tr1% (70% by default) of all edges
            df_tr_1 = df.sample(n=int(len(df)*tr1), random_state=seed)
            remainder = pd.concat([df, df_tr_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)
            # balancing the edges and non-edges
            min_length = np.min([len(df_tr_1[df_tr_1['goal'] == 0]), len(df_tr_1[df_tr_1['goal'] == 1])])
            df_tr_1 = pd.concat([\
                df_tr_1[df_tr_1['goal'] == 0].sample(n=min_length, random_state=seed),\
                df_tr_1[df_tr_1['goal'] == 1].sample(n=min_length, random_state=seed)],\
                ignore_index=True)
        else:
            # balancing the edges and non-edges
            min_length = np.min([len(df[df['goal'] == 0]), len(df[df['goal'] == 1])])
            df = pd.concat([\
                df[df['goal'] == 0].sample(n=min_length, random_state=seed),\
                df[df['goal'] == 1].sample(n=min_length, random_state=seed)],\
                ignore_index=True)
            # randomly choosing tr1% (70% by default) of all edges
            df_tr_1 = df.sample(n=int(len(df)*tr1), random_state=seed)
            remainder = pd.concat([df, df_tr_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        # randomly choosing ts1% (15% by default) of all edges
        df_ts_1 = remainder.sample(n=int(len(df)*ts1), random_state=seed, ignore_index=True)
        # the rest is test 2
        df_ts_2 = pd.concat([remainder, df_ts_1], ignore_index=True).drop_duplicates(keep=False, ignore_index=True)

        log.info('Success!')
        return df_tr_1, df_ts_1, df_ts_2


    def calculate_features(self, df, G, t_feature_names, features_to_normalize):
        
        log.info(f'Calculating topological features: {t_feature_names}')

        functions = {
            'RAI': nx.resource_allocation_index,
            'JC': nx.jaccard_coefficient,
            'AAI': nx.adamic_adar_index,
            'PA': nx.preferential_attachment
        }

        for feature in t_feature_names:
            frm = df['node1'].values
            to = df['node2'].values
            edges = [(frm[i], to[i]) for i in range(len(frm))]
            features = [f[2] for f in functions[feature](G, edges)]
            
            df[feature] = pd.Series(features)
            if feature in features_to_normalize:
                df[feature] -= df[feature].min()
                df[feature] /= df[feature].max()

        log.info('Done calculating topological features')

        return df