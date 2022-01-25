from data_generators import generator

from SETTINGS import *

from models.model import model
from data.data_arrange import data_arrange
from plotting.plotting_funcs import *

from tensorflow.keras.utils import plot_model

import pandas as pd

import math


class Experiment:
    def __init__(self, generator: generator) -> None:
        self.generator = generator

    def run(self, exp_num):
        self.generator.load_data()
        data = self.generator.get_data()

        data['train_1'].dropna(inplace=True)
        data['test_1'].dropna(inplace=True)
        data['test_2'].dropna(inplace=True)

        attr_dim = len(data['train_1'].head(1)['node_1_attrs'].values[0]) if USE_ATTRIBUTES else 1

        link_prediction_model = model(TOPOLOGICAL_FEATURE_NAMES, attr_dim, embed_dim=EMBED_DIM, attrs=USE_ATTRIBUTES)
        plot_model(link_prediction_model, show_shapes=True,
                   to_file=RESULT_PATH + str(exp_num) + '/Link prediction model.png')

        x, y = data_arrange(data['train_1'], TOPOLOGICAL_FEATURE_NAMES, attrs=USE_ATTRIBUTES)
        link_prediction_model.fit(x, y, epochs=EPOCH)

        feature_importance(link_prediction_model, TOPOLOGICAL_FEATURE_NAMES, embed_dim=EMBED_DIM, attrs=USE_ATTRIBUTES,
                           path=RESULT_PATH + str(exp_num) + '/Feature importance of link prediction model.png')

        x, _ = data_arrange(data['test_1'], TOPOLOGICAL_FEATURE_NAMES, train=False, attrs=USE_ATTRIBUTES)

        prob = link_prediction_model.predict(x).squeeze()
        prob = pd.Series(prob, name='prob')

        link_proba = data['test_1'].join(prob)
        plot_auc(link_proba, x='goal', y='prob', path=RESULT_PATH + str(exp_num) + '/AUC of link prediction model.png')

        link_proba['abs_error'] = link_proba.apply(lambda row: math.fabs(row['goal'] - row['prob']), axis=1)
        train_median_error = link_proba['abs_error'].median()

        link_proba['quality_label'] = link_proba.apply(lambda row: 1 if row['abs_error'] <= train_median_error else 0,
                                                       axis=1)

        classification_model = model(TOPOLOGICAL_FEATURE_NAMES, attr_dim, attrs=USE_ATTRIBUTES)
        plot_model(classification_model, show_shapes=True,
                   to_file=RESULT_PATH + str(exp_num) + '/Classification model.png')

        x, y = data_arrange(link_proba, TOPOLOGICAL_FEATURE_NAMES, goal='quality_label', attrs=USE_ATTRIBUTES)
        classification_model.fit(x, y, epochs=EPOCH)

        feature_importance(classification_model, TOPOLOGICAL_FEATURE_NAMES, embed_dim=EMBED_DIM, attrs=USE_ATTRIBUTES,
                           path=RESULT_PATH + str(exp_num) + '/Feature importance of classification model.png')

        T = 0.5

        x, _ = data_arrange(data['test_2'], TOPOLOGICAL_FEATURE_NAMES, train=False, attrs=USE_ATTRIBUTES)

        quality_probability = classification_model.predict(x).squeeze()
        link_probability = link_prediction_model.predict(x).squeeze()

        quality_label = [1 if i > T else 0 for i in quality_probability]

        data['test_2'] = data['test_2'].join(pd.Series(quality_probability, name='predicted_quality_prob'))
        data['test_2'] = data['test_2'].join(pd.Series(link_probability, name='predicted_link_probability'))
        data['test_2'] = data['test_2'].join(pd.Series(quality_label, name='predicted_quality_label'))

        data['test_2']['true_abs_error'] = data['test_2'].apply(
            lambda row: math.fabs(row['goal'] - row['predicted_link_probability']), axis=1)
        test_median_error = data['test_2']['true_abs_error'].median()

        data['test_2']['true_quality_label'] = data['test_2'].apply(
            lambda row: 1 if row['true_abs_error'] <= test_median_error else 0, axis=1)

        plot_auc(data['test_2'], 'true_quality_label', 'predicted_quality_prob',
                 path=RESULT_PATH + str(exp_num) + '/AUC of classification model.png')
