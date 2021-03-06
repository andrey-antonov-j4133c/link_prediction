import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import modin.pandas as pd

from sklearn.metrics import roc_curve, auc

from settings import *


def plot_auc(df, x='goal', y='prob', path=None):
    mpl.rcParams['figure.figsize'] = [8, 5]
    mpl.rcParams['figure.dpi'] = 125
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots()

    fpr, tpr, _ = roc_curve(df[x].astype(int), df[y].astype(float))
    sns.lineplot(x=fpr, y=tpr, ax=ax)

    ax.legend([f'ROC curve, AUC = {auc(fpr, tpr):.4f}'])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    if path:
        pdf = PdfPages(path)
        pdf.savefig(fig)
        pdf.close()


def feature_importance(top_important_features, name, path=None):
    mpl.rcParams['figure.figsize'] = [12, 10]
    mpl.rcParams['figure.dpi'] = 125
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots()

    top_important_features = top_important_features.iloc[:FEATURE_IMPORTANCE_CUTOFF]
    clrs = ['lightcoral' if x in TOPOLOGICAL_FEATURE_NAMES else 'lightskyblue' for x in top_important_features.index]

    sns.barplot(
        top_important_features.index,
        top_important_features.values,
        ax=ax,
        palette=clrs,
        alpha=1
    )

    ax.set_title("SHAP Feature importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("SHAP Values")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend()

    if path:
        pdf = PdfPages(path)
        pdf.savefig(fig)
        pdf.close()


def feature_distribution(feature_df, features, goal='true_quality_label', path=None):
    def get_range(df, f):
        return df[f].max() - df[f].min()

    feature_df = feature_df[list(features) + [goal]]

    good_predictability = feature_df[feature_df[goal] == 1]
    bad_predictability = feature_df[feature_df[goal] == 0]

    mpl.rcParams['font.size'] = 14

    figures = []
    for feature in features:
        figure = plt.figure()
        ax = figure.subplots()

        if good_predictability[feature].min() > 0:
            good_predictability[feature] -= good_predictability[feature].min()

        if bad_predictability[feature].min() > 0:
            bad_predictability[feature] -= bad_predictability[feature].min()

        sns.histplot(
            good_predictability,
            x=feature,
            binwidth=get_range(feature_df, feature) / 15,
            stat='density',
            ax=ax,
            color='lightcoral',
            alpha=1,
            log_scale=False)

        sns.histplot(
            bad_predictability,
            x=feature,
            binwidth=get_range(feature_df, feature) / 15,
            stat='density',
            ax=ax,
            color='lightskyblue',
            alpha=0.85,
            log_scale=False)

        ax.set_ylabel("Density")
        ax.set_yscale('log')

        figures.append(figure)

    if path:
        pdf = PdfPages(path)
        for figure in figures:
            pdf.savefig(figure)
        pdf.close()
