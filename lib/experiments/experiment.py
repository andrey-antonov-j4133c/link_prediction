import math
import os

import modin.pandas as pd

from utils.data_arrange import data_arrange, add_attributes
from utils.metrics import calculate_metrics

from data_formatting import formatter
from plotting.plotting_funcs import *


class Experiment:
    def __init__(self, generator: formatter, model) -> None:
        self.generator = generator
        self.model_cls = model

    def get_attributes(self, dfs, attrs_dict, attr_dim):
        for i in range(len(dfs)):
            for j in range(attr_dim):
                dfs[i][f'node_1_attr_{j}'] = dfs[i]['node1'].apply(lambda x: attrs_dict[x][j])
                dfs[i][f'node_2_attr_{j}'] = dfs[i]['node2'].apply(lambda x: attrs_dict[x][j])

        return dfs

    def classify(self,
                 train_df: pd.DataFrame,
                 train_y: str,
                 test_df: pd.DataFrame,
                 test_y: str,
                 features: list,
                 attr_dim, attributed, random_state, path, i, all_features=False):

        model_name = f'cls_model_{i}'
        path += f'/{model_name}/'
        os.makedirs(RESULT_PATH + path)

        pd.DataFrame({"Features used": features}).to_csv(RESULT_PATH + path + f'/features_used.csv')

        model = self.model_cls(
            feature_cols=TOPOLOGICAL_FEATURE_NAMES if all_features else features,
            name=model_name,
            args={'attr_dim': attr_dim, 'attributed': attributed, 'random_state': random_state},
            type='full' if all_features else 'selected_features')

        model.plot_model(path=path)

        model.fit(train_df, train_y)

        feature_importance = None
        if all_features:
            feature_importance = model.feature_importance(
                train_df.sample(n=FI_SAMPLES, replace=False),
                test_df.sample(n=FI_SAMPLES, replace=False),
                path=path)

            feature_distribution(
                train_df,
                feature_importance.index.values[:FEATURE_IMPORTANCE_CUTOFF],
                goal='quality_label',
                path=RESULT_PATH + path + f'/feature_dist.pdf'
            )

        quality_prob = model.predict(test_df, path)
        test_df = test_df.join(pd.Series(quality_prob, name=f'pred_quality_prob_{model_name}'))
        test_df = test_df.join(
            pd.Series([1 if i > 0.5 else 0 for i in quality_prob], name=f'pred_quality_label_{model_name}'))

        plot_auc(
            test_df,
            'true_quality_label',
            f'pred_quality_label_{model_name}',
            path=RESULT_PATH + path + '/auc.pdf')

        metrics = calculate_metrics(
            test_df['true_quality_label'].values,
            test_df[f'pred_quality_label_{model_name}'].values,
            model_name,
            ['average_precision', 'roc_auc']
        )

        return model, feature_importance, model_name, metrics

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

        metrics = pd.DataFrame([
            link_prediction_metrics,
            all_features_metrics])

        metrics.to_csv(RESULT_PATH + path + '/results.csv')
