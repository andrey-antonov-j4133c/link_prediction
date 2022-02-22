import math
import os

import modin.pandas as pd

from data_formatting import formatter
from experiments.experiment import Experiment
from experiments.feature_selection_experiment import FeatureSelectionExperiment

from utils.metrics import calculate_metrics
from plotting.plotting_funcs import plot_auc, feature_distribution

from settings import *


class VaryingFeatureSelectionExperiment(FeatureSelectionExperiment):
    def __init__(self, generator: formatter, model):
        super().__init__(generator, model)

    def run(self, attributed, path, random_state=0):
        train_1, test_1, test_2, attributes = self.generator.load_data()

        attr_dim = len(attributes.head(1)['attrs'].values[0]) if attributed else 1

        if attributed:
            train_1, test_1, test_2 = self.get_attributes(
                [train_1, test_1, test_2],
                attributes['attrs'].to_dict(),
                attr_dim
            )

        # PART 2 -- models invocation
        link_prediction_model = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES,
            name='Link prediction',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state})

        link_prediction_model.plot_model(path=path)

        # PART 3 -- link prediction model fitting
        link_prediction_model.fit(train_1, 'goal')

        # link_prediction_model.feature_importance(
        #    train_1.sample(n=FI_SAMPLES, replace=False),
        #    train_1.sample(n=FI_SAMPLES, replace=False),
        #    path=path)

        # PART 4 -- predicting links
        prob = link_prediction_model.predict(test_1)
        prob = pd.Series(prob, name='prob')

        link_probability = test_1.join(prob)
        plot_auc(
            link_probability,
            x='goal', y='prob',
            path=RESULT_PATH + path + '/AUC of link prediction model.pdf')

        link_prediction_metrics = calculate_metrics(
            link_probability['goal'].values,
            link_probability['prob'].values,
            'Link prediction',
            ['average_precision', 'roc_auc']
        )

        link_probability['abs_error'] = link_probability.apply(lambda row: math.fabs(row['goal'] - row['prob']), axis=1)
        train_median_error = link_probability['abs_error'].median()

        link_probability['quality_label'] = link_probability.apply(
            lambda row: 1 if row['abs_error'] <= train_median_error else 0, axis=1)

        T = 0.5

        link_proba = link_prediction_model.predict(test_2)
        test_2 = test_2.join(pd.Series(link_proba, name='predicted_link_probability'))

        test_2['true_abs_error'] = test_2.apply(
            lambda row: math.fabs(row['goal'] - row['predicted_link_probability']), axis=1)
        test_median_error = test_2['true_abs_error'].median()

        test_2['true_quality_label'] = test_2.apply(
            lambda row: 1 if row['true_abs_error'] <= test_median_error else 0, axis=1)

        cols = TOPOLOGICAL_FEATURE_NAMES.copy()
        if attributed:
            cols += [f'node_1_attr_{i}' for i in range(attr_dim)]
            cols += [f'node_2_attr_{i}' for i in range(attr_dim)]

        classification_model_all_features, \
            feature_importance, \
            all_features_name, \
            all_features_metrics = self.classify(
                link_probability, 'quality_label', test_2, 'true_quality_label',
                cols, attr_dim, attributed, random_state, path, 0, all_features=True)

        metrics_arr = [link_prediction_metrics, all_features_metrics]

        i = 1
        to = 2
        while to < len(feature_importance):
            features = feature_importance.iloc[:to]

            classification_model_selected_features, \
                _, \
                selected_features_name, \
                selected_features_metrics = self.classify(
                    link_probability, 'quality_label', test_2, 'true_quality_label',
                    features.index.values, attr_dim, attributed, random_state, path, i, all_features=False)

            metrics_arr.append(selected_features_metrics)
            i += 1
            to += 2

        metrics = pd.DataFrame(metrics_arr)
        metrics.to_csv(RESULT_PATH + path + '/results.csv')
