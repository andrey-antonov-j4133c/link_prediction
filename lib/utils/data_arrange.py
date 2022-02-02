import numpy as np

from settings import *


def data_arrange(df, train=True, attrs=True, goal='goal'):
    x_features = df[TOPOLOGICAL_FEATURE_NAMES].values

    x_attrs_1 = np.array([np.array(item) for item in df['node_1_attrs'].values]) if attrs else []
    x_attrs_2 = np.array([np.array(item) for item in df['node_2_attrs'].values]) if attrs else []

    y = df[goal].values if train else None

    if attrs:
        return [x_features, x_attrs_1, x_attrs_2], y
    else:
        return x_features, y
