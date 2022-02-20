import os
import pandas as pd

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from settings import *


def plot_results(df, path=None):
    mpl.rcParams['figure.figsize'] = [8, 5]
    mpl.rcParams['figure.dpi'] = 125

    fig, ax = plt.subplots()

    sns.lineplot(data=df, x='Alpha', y="Link prediction AP", ax=ax, color='lightcoral', label='Link prediction average precision')
    sns.lineplot(data=df, x='Alpha', y="Link prediction AUC", ax=ax, color='lightskyblue', label='Link prediction ROC-AUC')
    sns.lineplot(data=df, x='Alpha', y="Classification AP", ax=ax, color='tan', label='Classification average precision')
    sns.lineplot(data=df, x='Alpha', y="Classification AUC", ax=ax, color='mediumseagreen', label='Classification ROC-AUC')

    ax.set_ylabel("Metrics")
    ax.set_xlabel("Alpha")

    if path:
        pdf = PdfPages(path)
        pdf.savefig(fig)
        pdf.close()


def main():
    results = {
        "Alpha": [],
        "Link prediction AP": [],
        "Link prediction AUC": [],
        "Classification AP": [],
        "Classification AUC": []
    }

    for dir in os.listdir(RESULT_PATH):
        if dir[:6] != "acMark":
            continue
        alpha = float(dir.split(';')[0].replace('acMark-a=', ''))
        results_pd = pd.read_csv(RESULT_PATH + dir + '/results.csv')

        results['Alpha'].append(alpha)
        results['Link prediction AP'].append(results_pd.iloc[0]['average_precision'])
        results['Link prediction AUC'].append(results_pd.iloc[0]['roc_auc'])
        results['Classification AP'].append(results_pd.iloc[1]['average_precision'])
        results['Classification AUC'].append(results_pd.iloc[1]['roc_auc'])

    df = pd.DataFrame(results).sort_values(by='Alpha')
    plot_results(df, RESULT_PATH + "_Additional plots/acMark.pdf")


if __name__ == '__main__':
    main()