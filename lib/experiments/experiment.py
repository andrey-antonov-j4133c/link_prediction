import math

import pandas as pd

from utils.data_arrange import data_arrange
from data_formatting import formatter
from plotting.plotting_funcs import *


class Experiment:
    def __init__(self, generator: formatter, model) -> None:
        self.generator = generator
        self.model_cls = model

    def run(self, attributed, path):
        data = self.generator.load_data()

        data['train_1'].dropna(inplace=True)
        data['test_1'].dropna(inplace=True)
        data['test_2'].dropna(inplace=True)

        attr_dim = len(data['train_1'].head(1)['node_1_attrs'].values[0]) if attributed else 1

        link_prediction_model = self.model_cls(
            name='Link prediction',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': 0})
        link_prediction_model.plot_model(path=path)

        x, y = data_arrange(data['train_1'], TOPOLOGICAL_FEATURE_NAMES, attrs=attributed)

        link_prediction_model.fit(x, y)
        link_prediction_model.feature_importance(path=path)

        x, _ = data_arrange(data['test_1'], TOPOLOGICAL_FEATURE_NAMES, train=False, attrs=attributed)

        prob = link_prediction_model.predict(x)
        prob = pd.Series(prob, name='prob')

        link_proba = data['test_1'].join(prob)
        plot_auc(link_proba, x='goal', y='prob', path=RESULT_PATH + path + '/AUC of link prediction model.png')

        link_proba['abs_error'] = link_proba.apply(lambda row: math.fabs(row['goal'] - row['prob']), axis=1)
        train_median_error = link_proba['abs_error'].median()

        link_proba['quality_label'] = link_proba.apply(
            lambda row: 1 if row['abs_error'] <= train_median_error else 0, axis=1)

        classification_model = self.model_cls(
            name='Classification',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': 0})
        classification_model.plot_model(path=path)

        x, y = data_arrange(link_proba, TOPOLOGICAL_FEATURE_NAMES, goal='quality_label', attrs=attributed)
        classification_model.fit(x, y)
        classification_model.feature_importance(path=path)

        T = 0.5

        x, _ = data_arrange(data['test_2'], TOPOLOGICAL_FEATURE_NAMES, train=False, attrs=attributed)

        quality_probability = classification_model.predict(x)
        link_probability = link_prediction_model.predict(x)

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
                 path=RESULT_PATH + path + '/AUC of classification model.png')
