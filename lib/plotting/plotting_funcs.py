import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import roc_curve, auc

import tensorflow.keras.backend as K
import numpy as np


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


def feature_importance(model, feature_names, embed_dim, layer_name='Hidden_layer', attrs=True, path=None):
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
        zero_tensor = K.constant(np.zeros((1, len(feature_names) + embed_dim)))
    else:
        zero_tensor = K.constant(np.zeros((1, len(feature_names))))

    if attrs:
        x = feature_names + ['{}'.format(i + 1) for i in range(embed_dim)]
    else:
        x = feature_names
    y = K.eval(layer(zero_tensor))

    if attrs:
        mpl.rcParams['figure.figsize'] = [15, 8]
    else:
        mpl.rcParams['figure.figsize'] = [6, 4]
    mpl.rcParams['figure.dpi'] = 125

    plot(x, y[0])
