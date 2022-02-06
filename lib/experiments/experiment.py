import math

import modin.pandas as pd

from utils.data_arrange import data_arrange, add_attributes
from utils.metrics import calculate_metrics

from data_formatting import formatter
from plotting.plotting_funcs import *


class Experiment:
    def __init__(self, generator: formatter, model) -> None:
        self.generator = generator
        self.model_cls = model

    def run(self, attributed, path, random_state=0):
        # PART 1 -- data reading and preparation
        train_1, test_1, test_2, attributes = self.generator.load_data()

        attr_dim = len(attributes.head(1)['attrs'].values[0]) if attributed else 1

        if attributed:
            attributes = attributes['attrs'].to_dict()

            train_1 = add_attributes(train_1, attributes, attr_dim)
            test_1 = add_attributes(test_1, attributes, attr_dim)
            test_2 = add_attributes(test_2, attributes, attr_dim)

        # PART 2 -- models invocation
        link_prediction_model = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES,
            name='Link prediction',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state})

        classification_model = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES,
            name='Classification',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state})

        link_prediction_model.plot_model(path=path)
        classification_model.plot_model(path=path)

        # PART 3 -- link prediction model fitting
        link_prediction_model.fit(train_1, 'goal')
        #link_prediction_model.feature_importance(train_1.sample(n=FI_SAMPLES, replace=False), path=path)

        # PART 4 -- predicting links
        prob = link_prediction_model.predict(test_1)
        prob = pd.Series(prob, name='prob')

        link_probability = test_1.join(prob)
        plot_auc(link_probability, x='goal', y='prob', path=RESULT_PATH + path + '/AUC of link prediction model.png')

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

        classification_model.fit(link_probability, 'quality_label')
        classification_model.feature_importance(link_probability.sample(n=FI_SAMPLES, replace=False), path=path)

        T = 0.5

        link_probability = link_prediction_model.predict(test_2)
        quality_probability = classification_model.predict(test_2)

        quality_label = [1 if i > T else 0 for i in quality_probability]

        test_2 = test_2.join(pd.Series(quality_probability, name='predicted_quality_prob'))
        test_2 = test_2.join(pd.Series(link_probability, name='predicted_link_probability'))
        test_2 = test_2.join(pd.Series(quality_label, name='predicted_quality_label'))

        test_2['true_abs_error'] = test_2.apply(
            lambda row: math.fabs(row['goal'] - row['predicted_link_probability']), axis=1)
        test_median_error = test_2['true_abs_error'].median()

        test_2['true_quality_label'] = test_2.apply(
            lambda row: 1 if row['true_abs_error'] <= test_median_error else 0, axis=1)

        plot_auc(test_2, 'true_quality_label', 'predicted_quality_prob',
                 path=RESULT_PATH + path + '/AUC of classification model.png')

        classification_metrics = calculate_metrics(
            test_2['true_quality_label'].values,
            test_2['predicted_quality_prob'].values,
            'Classification',
            ['average_precision', 'roc_auc']
        )

        metrics = pd.DataFrame([link_prediction_metrics, classification_metrics])
        metrics.to_csv(RESULT_PATH + path + '/results.csv')
