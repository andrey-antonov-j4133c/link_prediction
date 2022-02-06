import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import tensorflow.keras.backend as K
from sklearn.metrics import roc_curve, auc

from settings import *


def plot_auc(df, x='goal', y='prob', path=None):
    mpl.rcParams['figure.figsize'] = [8, 5]
    mpl.rcParams['figure.dpi'] = 125

    _, ax = plt.subplots(1)

    fpr, tpr, _ = roc_curve(df[x], df[y])
    sns.lineplot(x=fpr, y=tpr, ax=ax)

    ax.legend([f'ROC curve, AUC = {auc(fpr, tpr):.4f}'])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    if path:
        plt.savefig(path)
        plt.clf()


def feature_importance_keras(model, layer_name='Hidden_layer', attrs=True, path=None):
    def plot(x, y):
        sns.barplot(
            x=x,
            y=y,
            label='Reconstruction feature importance',
            color='lightcoral',
            alpha=1
        )

        plt.xlabel('Feature')
        plt.ylabel('NN Wheight')

        plt.title('Relative feature importance')

        if path:
            plt.savefig(path)
            plt.clf()

    layer = model.get_layer(layer_name)
    if attrs:
        zero_tensor = K.constant(np.zeros((1, len(TOPOLOGICAL_FEATURE_NAMES) + EMBED_DIM)))
    else:
        zero_tensor = K.constant(np.zeros((1, len(TOPOLOGICAL_FEATURE_NAMES))))

    if attrs:
        x = TOPOLOGICAL_FEATURE_NAMES + ['{}'.format(i + 1) for i in range(EMBED_DIM)]
    else:
        x = TOPOLOGICAL_FEATURE_NAMES
    y = K.eval(layer(zero_tensor))

    if attrs:
        mpl.rcParams['figure.figsize'] = [15, 8]
    else:
        mpl.rcParams['figure.figsize'] = [6, 4]
    mpl.rcParams['figure.dpi'] = 125

    plot(x, y[0])


def feature_importance(top_important_features, name, path=None):
    mpl.rcParams['figure.figsize'] = [6, 4]
    mpl.rcParams['figure.dpi'] = 125

    fig, ax = plt.subplots()

    sns.barplot(
        top_important_features.index,
        top_important_features.values,
        label='Feature importance',
        ax=ax,
        color='lightcoral',
        alpha=1
    )

    ax.set_title("SHAP Feature importance")
    ax.set_ylabel("SHAP Values")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend()

    if path:
        plt.savefig(path)
        plt.clf()
