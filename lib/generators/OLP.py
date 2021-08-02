import pickle
from typing import Generator  
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
import scipy.sparse as sparse

from lib.generators.generator_base import Generator
from lib.generators.OLP_ import *

class OLP(Generator):
    def __init__(
            self, 
            feature_names = ['com_ne', 'short_path', 'LHN', 'page_rank_pers_edges', 'pref_attach', 'jacc_coeff', 'adam_adar', 'res_alloc_ind', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean'],
            features_to_normalize = ['com_ne', 'short_path', 'pref_attach', 'svd_edges_dot'],
            network_index=0, seed=0) -> None:

        self.feature_names = feature_names

        print('Loading file...')
        infile = open('./data/OLP/OLP_updated.pickle','rb')  
        networks_df = pickle.load(infile)
        networks_df = networks_df.sort_values(by=['number_edges'], ascending=False)
        df_edgelists = networks_df['edges_id']
        edges_orig = df_edgelists.iloc[network_index]
        print('File loaded!')

        print('Calculating features...')
        features_df = self.calc_features(edges_orig, features_to_normalize)
        print('Features calculated!')

        print('Train/test splitting...')
        self.TRAIN1_DF, self.TEST1_DF, self.TEST2_DF = self.sample_edges(features_df, seed)
        print('Done splitting the data!')



    def calc_features(self, edges_orig, features_to_normalize):
        edges_orig = np.array(np.matrix(edges_orig))
        num_nodes = int(np.max(edges_orig)) + 1
        row = np.array(edges_orig)[:,0]
        col = np.array(edges_orig)[:,1]

        data_aux = np.ones(len(row))
        A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
        A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
        A_orig[A_orig>0] = 1 
        A_orig = A_orig.todense()

        #### construct the holdout and training matriced from the original matrix
        alpha = 0.8 # sampling rate for holdout matrix
        alpha_ = 0.8 # sampling rate for training matrix
        A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)

        #### extract features #####
        sample_true_false_edges(A_orig, A_tr, A_ho)
        edge_t_tr = np.loadtxt("./data/OLP/edge_tf_tr/edge_t.txt").astype('int')
        edge_f_tr = np.loadtxt("./data/OLP/edge_tf_tr/edge_f.txt").astype('int')
        df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
        df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)

        #### load dataframes for train and holdout sets ####
        df_tr = creat_full_set(df_t_tr,df_f_tr)

        ground_truth_df = df_tr[['i', 'j', 'TP'] + self.feature_names]
        ground_truth_df = ground_truth_df.rename(columns={'i': 'node1', 'j': 'node2', 'TP': 'goal'})

        n = np.min(ground_truth_df['goal'].value_counts())

        ground_truth_df = ground_truth_df[['node1', 'node2', 'goal']]

        ground_truth_df0 = ground_truth_df[ground_truth_df['goal'] == 0]
        ground_truth_df1 = ground_truth_df[ground_truth_df['goal'] == 1]

        df = pd.concat([ground_truth_df0, ground_truth_df1], ignore_index=True)

        features_df = df[['node1', 'node2', 'goal']].merge(
            df_tr[['i', 'j'] + self.feature_names].rename(columns={'i': 'node1', 'j': 'node2'}),
            how='inner'
        )

        # normalize
        for column in features_to_normalize:
            features_df[column] = (features_df[column]-features_df[column].min())/(features_df[column].max()-features_df[column].min())
        
        return features_df

    def sample_edges(self, df, seed, tr1=0.7, ts1=0.15):
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