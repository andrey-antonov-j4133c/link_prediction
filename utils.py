from math import e
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def convert_nx_to_dataframe(G, Compl_G, n_samples=500):
    df = pd.DataFrame(list(G.edges()), columns=['node1', 'node2'])
    df[['goal']] = 1
    comp_df = pd.DataFrame(list(Compl_G.edges()), columns=['node1', 'node2'])
    comp_df[['goal']] = 0
    df = df.append(comp_df)

    return df


def top_k(classifier, feature_names, k, test_df):
    prob = classifier.predict_proba(test_df[feature_names])[:, 1]
    prob = pd.Series(prob, name='prob')
    result = test_df.join(prob)

    top = result[['node1', 'node2', 'prob']].nlargest(k, 'prob')

    X1 = top.join(test_df[feature_names])
    X2 = top.join(test_df['goal'])

    score = classifier.score(X1[feature_names], X2['goal'])

    d0 = {}
    d1 = {}

    for element, goal in zip(sum(X1[['prob']].T.values.tolist(), []), sum(X2[['goal']].T.values.tolist(),[])):
        if goal == 0:
            if element in d0.keys(): d0[element] += 1
            else: d0[element] = 1
        elif goal == 1:
            if element in d1.keys(): d1[element] += 1
            else: d1[element] = 1

    df0 = pd.DataFrame(list(d0.items()), columns = ['prob', 'count'])
    df1 = pd.DataFrame(list(d1.items()), columns = ['prob', 'count'])


    return score, df0, df1, top[['node1', 'node2', 'prob']].merge(test_df[['node1', 'node2', 'goal']], on=['node1','node2'])


def plot_bars(df0, df1, bins):
    pd.cut(df0['prob'], bins=bins, include_lowest=True).value_counts(sort=False).plot.bar(rot=0, color='steelblue', alpha=.5, label='0')
    pd.cut(df1['prob'], bins=bins, include_lowest=True).value_counts(sort=False).plot.bar(rot=0, color='firebrick', alpha=.5, label='1')
    plt.legend()
