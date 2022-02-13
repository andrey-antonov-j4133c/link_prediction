import math

import modin.pandas as pd

from data_formatting import formatter
from experiments.experiment import Experiment

from utils.metrics import calculate_metrics
from plotting.plotting_funcs import plot_auc, feature_distribution

from settings import *


class FeatureSelectionExperiment(Experiment):
    def __init__(self, generator: formatter, model):
        super().__init__(generator, model)

    def re_sample_edges(self, train_1, test_1, test_2, tr1=0.1, ts1=0.05, ts2=0.1):
        def balance_data(df):
            min_length = min([len(df[df['goal'] == 1]), len(df[df['goal'] == 0])])
            return pd.concat([
                df[df['goal'] == 1].sample(n=min_length, replace=False),
                df[df['goal'] == 0].sample(n=min_length, replace=False),
            ]).reset_index(drop=True)

        df = pd.concat([train_1, test_1, test_2]).reset_index(drop=True)

        train_1 = df.sample(n=int(len(df)*tr1)).reset_index(drop=True)
        train_1 = balance_data(train_1)

        remainder = pd.concat([df, train_1]).drop_duplicates(keep=False).reset_index(drop=True)

        test_1 = remainder.sample(n=int(len(df) * ts1)).reset_index(drop=True)

        remainder = pd.concat([remainder, test_1], ignore_index=True).drop_duplicates(keep=False).reset_index(drop=True)

        remainder = pd.concat([remainder, test_2], ignore_index=True).drop_duplicates(keep=False).reset_index(drop=True)

        return train_1, test_1, remainder

    def run(self, attributed, path, random_state=0):
        train_1, test_1, test_2, attributes = self.generator.load_data()
        train_1, test_1, remainder = self.re_sample_edges(train_1, test_1, test_2)

        attr_dim = len(attributes.head(1)['attrs'].values[0]) if attributed else 1

        if attributed:
            train_1, test_1, remainder = self.get_attributes(
                [train_1, test_1, remainder],
                attributes['attrs'].to_dict(),
                attr_dim
            )

        # PART 2 -- models invocation
        link_prediction_model = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES,
            name='Link prediction',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state})

        classification_model_all_features = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES,
            name='Classification on all features',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state})

        link_prediction_model.plot_model(path=path)
        classification_model_all_features.plot_model(path=path)

        # PART 3 -- link prediction model fitting
        link_prediction_model.fit(train_1, 'goal')

        #link_prediction_model.feature_importance(
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

        classification_model_all_features.fit(link_probability, 'quality_label')
        classification_feature_importance = classification_model_all_features.feature_importance(
            link_probability.sample(n=FI_SAMPLES, replace=False),
            remainder.sample(n=FI_SAMPLES, replace=False),
            path=path)

        feature_distribution(
            link_probability,
            classification_feature_importance.index.values,
            goal='quality_label',
            path=RESULT_PATH + path + '/Top features distribution.pdf'
        )

        most_important_features = classification_feature_importance.index.values

        classification_model_selected_features = self.model_cls(
            feature_cols=most_important_features,
            name='Classification on selected features',
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state},
            type='selected_features')

        classification_model_selected_features.fit(link_probability, 'quality_label')

        T = 0.5

        link_probability = link_prediction_model.predict(remainder)
        remainder = remainder.join(pd.Series(link_probability, name='predicted_link_probability'))

        remainder['true_abs_error'] = remainder.apply(
            lambda row: math.fabs(row['goal'] - row['predicted_link_probability']), axis=1)
        test_median_error = remainder['true_abs_error'].median()

        remainder['true_quality_label'] = remainder.apply(
            lambda row: 1 if row['true_abs_error'] <= test_median_error else 0, axis=1)

        quality_probability_all_features = classification_model_all_features.predict(remainder)
        remainder = remainder.join(pd.Series(
            quality_probability_all_features,
            name='predicted_quality_prob_all_features'))
        remainder = remainder.join(
            pd.Series([1 if i > T else 0 for i in quality_probability_all_features],
            name='predicted_quality_label_all_features'))

        plot_auc(
            remainder,
            'true_quality_label',
            'predicted_quality_prob_all_features',
            path=RESULT_PATH + path + '/AUC of classification model with all features.pdf')

        classification_metrics_all_features = calculate_metrics(
            remainder['true_quality_label'].values,
            remainder['predicted_quality_prob_all_features'].values,
            'Classification all features',
            ['average_precision', 'roc_auc']
        )

        quality_probability_selected_features = classification_model_selected_features.predict(remainder)
        remainder = remainder.join(pd.Series(
            quality_probability_selected_features,
            name='predicted_quality_prob_selected_features'))
        remainder = remainder.join(
            pd.Series([1 if i > T else 0 for i in quality_probability_selected_features],
                      name='predicted_quality_label_selected_features'))

        plot_auc(
            remainder,
            'true_quality_label',
            'predicted_quality_prob_selected_features',
            path=RESULT_PATH + path + '/AUC of classification model with selected features.pdf')

        classification_metrics_selected_features = calculate_metrics(
            remainder['true_quality_label'].values,
            remainder['predicted_quality_prob_selected_features'].values,
            'Classification selected features',
            ['average_precision', 'roc_auc']
        )

        metrics = pd.DataFrame([
            link_prediction_metrics,
            classification_metrics_all_features,
            classification_metrics_selected_features])

        metrics.to_csv(RESULT_PATH + path + '/results.csv')
