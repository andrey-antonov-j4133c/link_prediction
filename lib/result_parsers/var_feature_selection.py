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
    mpl.rcParams['font.size'] = 14

    fig, ax = plt.subplots()
    ax.set_xticks(range(0, df['Features used'].max(), int(df['Features used'].max()/10)))
    ax2 = ax.twinx()

    sns.lineplot(data=df, x='Features used', y="Classification AP", ax=ax, color='lightskyblue', label='Average precision')
    sns.lineplot(data=df, x='Features used', y="Time", ax=ax2, color='lightcoral', label='Time')

    ax.set_ylabel("Average Precision")
    ax2.set_ylabel("Time in seconds")
    ax.set_xlabel("Number of features used")

    ax.legend(loc='lower right')
    ax2.legend(loc='upper left')

    if path:
        pdf = PdfPages(path)
        pdf.savefig(fig)
        pdf.close()


def main():
    if not os.path.isdir(RESULT_PATH + '/_Additional_plots/'):
        os.makedirs(RESULT_PATH + '/_Additional_plots/')

    results_dict = {
        "Classification AP": [],
        "Classification AUC": [],
        "Time": [],
        "Features used": []
    }

    dataset = 'acMark-a=0.5;b=0.1;s=0.1;o=0.1_run1'
    experiment = ', attributed is True, model: Keras NN model, exp: Varying feature selection'

    folder_name = dataset + experiment

    results = pd.read_csv(RESULT_PATH + folder_name + '/results.csv')
    for dir in os.listdir(RESULT_PATH + folder_name + '/'):
        if dir[:3] != 'cls':
            continue

        features_used = pd.read_csv(RESULT_PATH + folder_name + '/' + dir + '/features_used.csv')
        predict_time = pd.read_csv(RESULT_PATH + folder_name + '/' + dir + f'/predict_time_for_{dir}.csv')
        calc_time = pd.read_csv(DATA_PATH + 'pre_computed/' + dataset + '/test2_time.csv')

        time = predict_time['Predict time'].values[0]

        for feature in features_used['Features used'].values:
            if feature not in calc_time.keys().values:
                continue
            time += calc_time[feature].values[0]

        results_dict['Time'].append(time)
        results_dict['Features used'].append(len(features_used['Features used']))

        results_dict['Classification AP'].append(results[results['Unnamed: 0'] == dir]['average_precision'].values[0])
        results_dict['Classification AUC'].append(results[results['Unnamed: 0'] == dir]['roc_auc'].values[0])

    res_df = pd.DataFrame(results_dict)
    plot_results(res_df, RESULT_PATH + f"_Additional_plots/var_feature_selection_for_{dataset}.pdf")


if __name__ == '__main__':
    main()