import logging as log
import random

import networkx as nx
import numpy as np
import pandas as pd

from settings import *


class OptFormatter:
    def __init__(self, args: dict, attributed=True) -> None:
        self.A = None
        self.H = None
        self.y = None

        self.attributed = attributed

    def load_data(self) -> None:
        G = nx.convert_matrix.from_numpy_matrix(self.A)
        G.remove_edges_from(nx.selfloop_edges(G))
        CG = nx.complement(G)

        all_edges = list(G.edges) + list(CG.edges)

        x_topological = self._calculate_features(G, all_edges)
        y = np.concatenate([np.ones(len(G.edges)), np.zeros(len(CG.edges))])

        if self.attributed:
            train1, test1, test2 = self._sample_edges(x_topological, self.H, y)
        else:
            train1, test1, test2 = self._sample_edges(x_topological, None, y)

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


    def _sample_edges(self, features, attrs, y, tr1=0.7, ts1=0.15, seed=0):
        def make_data_from_indexes(indexes):
            data = []
            for i in indexes:
                if attrs is not None:
                    row = (features[i], attrs[i], y[i])
                else:
                    row = (features[i], y[i])

        indexes = list(range(features.shape[0]))

        train_1_indexes = random.sample(indexes, int(tr1*len(features)))
        remainder = [index for index in indexes if index not in train_1_indexes]

        train_1_edges = [index for index in train_1_indexes if y[index] == 1]
        train_1_non_edges = [index for index in train_1_indexes if y[index] == 0]

        min_length = min(len(train_1_edges), len(train_1_non_edges))

        train_1_indexes = np.concatenate([
            np.random.sample(train_1_edges, min_length),
            np.random.sample(train_1_non_edges, min_length)])

        test_1_indexes = random.sample(remainder, int(ts1*len(features)))
        test_2_indexes = [index for index in remainder if index not in test_1_indexes]

        return \
            make_data_from_indexes(train_1_indexes),\
            make_data_from_indexes(test_1_indexes),\
            make_data_from_indexes(test_2_indexes)

    def _calculate_features(self, G, edges):
        functions = {
            'RAI': nx.resource_allocation_index,
            'JC': nx.jaccard_coefficient,
            'AAI': nx.adamic_adar_index,
            'PA': nx.preferential_attachment
        }

        features_matrix = np.zeros((len(edges), len(functions.keys())), np.float64)

        for i, feature in enumerate(TOPOLOGICAL_FEATURE_NAMES):
            feature_values = np.array([f[2] for f in functions[feature](G, edges)], np.float64)

            if feature in TOPOLOGICAL_FEATURES_TO_NORMALIZE:
                feature_values -= feature_values.min()
                feature_values /= feature_values.max()

            features_matrix[:, i] = feature_values

        return features_matrix
