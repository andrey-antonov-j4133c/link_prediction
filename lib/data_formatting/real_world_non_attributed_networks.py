"""
Data and some portions of the code in this file are
sourced from https://github.com/Aghasemian/OptimalLinkPrediction
"""
import pickle

import numpy as np

from settings import OLP_PATH
from data_formatting.formatter import Formatter


class RealWorldNonAttrFormatter(Formatter):
    def __init__(self, args: dict, attributed) -> None:
        super().__init__(args, attributed)
        self.A = self._read_data(args)

    def _read_data(self, args: dict):
        dataset = args['dataset_name']

        infile = open(OLP_PATH, 'rb')
        networks_df = pickle.load(infile)

        networks_df = networks_df[networks_df['network_name'] == dataset].head(1)

        edge_list = networks_df['edges_id'].iloc[0]
        num_edges = int(networks_df['number_nodes'].iloc[0])

        return self._convert_to_adg_list(edge_list, num_edges)

    def _convert_to_adg_list(self, edge_list, number_of_edges):
        adj_matrix = np.zeros((number_of_edges, number_of_edges))

        for i, j in edge_list:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1

        return adj_matrix
