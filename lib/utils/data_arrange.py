import numpy as np

from settings import *


def data_arrange(df, train=True, goal='goal'):
    x_features = df[TOPOLOGICAL_FEATURE_NAMES]

    y = df[goal] if train else None

    return x_features, y


def add_attributes(df, attrs_dict, attr_dim):
    for i in range(attr_dim):
        df[f'node_1_attr_{i}'] = df['node1'].apply(lambda x: attrs_dict[x][i])
        df[f'node_2_attr_{i}'] = df['node2'].apply(lambda x: attrs_dict[x][i])
    return df
