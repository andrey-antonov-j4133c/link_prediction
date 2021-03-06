import time
from abc import ABC
from models.model_wrapper import ModelWrapper

import pandas as pd
import modin.pandas as pd_1
from sklearn.ensemble import RandomForestClassifier

from plotting.plotting_funcs import feature_importance
from settings import *

import shap


class GBModel(ModelWrapper, ABC):
    MODEL_TYPE = 'Gradient Boosting model'

    def __init__(self, feature_cols, name, args, type='full'):
        super().__init__(feature_cols, name, args, type)

        self.n_estimators = 100
        self.criterion = 'gini'
        self.max_depth = 3
        self.random_state = args['random_state']

        self.model = self.__init_model()

    def fit(self, node_df, y_col):
        self.model.fit(node_df[self.feature_cols].values, node_df[y_col].values)

    def predict(self, node_df, path=None):
        stat = time.time()
        res = self.model.predict_proba(node_df[self.feature_cols].values)[:, 1]
        end = time.time()
        if path:
            pd.DataFrame({"Predict time": [end - stat]}).to_csv(RESULT_PATH + path + '/' + f'predict_time_for_{self.name}.csv')
        return res

    def feature_importance(self, train_samples, test_samples, path):
        def f(X):
            return self.model.predict_proba(X)

        explainer = shap.KernelExplainer(f, train_samples[self.feature_cols].values)
        shap_values = explainer.shap_values(test_samples[self.feature_cols].values, nsamples=FI_PERMUTATIONS)

        importance_pd = pd.DataFrame(
            shap_values[0],
            columns=self.feature_cols)

        top_important_features = importance_pd.mean(axis=0).sort_values(ascending=False)
        top_important_features = top_important_features \
            .reindex(top_important_features.map(lambda x: x).abs().sort_values(ascending=False).index)

        feature_importance(
            top_important_features,
            self.name,
            path=RESULT_PATH + path + '/' + f'Feature importance for {self.name}.pdf')

        return top_important_features

    def plot_model(self, path):
        return

    def __init_model(self):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state)

        return model
