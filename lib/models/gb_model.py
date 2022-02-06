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

    def __init__(self, feature_cols, name, args):
        super().__init__(feature_cols, name, args)

        self.n_estimators = 100
        self.criterion = 'gini'
        self.max_depth = 3
        self.random_state = args['random_state']

        self.model = self.__init_model()

    def fit(self, node_df, y_col):
        self.model.fit(node_df[self.feature_cols].values, node_df[y_col].values)

    def predict(self, node_df):
        return self.model.predict_proba(node_df[self.feature_cols].values)[:, 1]

    def feature_importance(self, samples, path):
        def f(X):
            return self.model.predict_proba(X)

        explainer = shap.KernelExplainer(f, samples[self.feature_cols].values)
        shap_values = explainer.shap_values(samples[self.feature_cols].values, nsamples=100)

        importance_pd = pd.DataFrame(
            shap_values[0],
            columns=self.feature_cols)

        top_important_features = importance_pd.mean(axis=0).sort_values(ascending=False).head(TOP_K_FEATURES)

        feature_importance(
            top_important_features,
            self.name,
            path=RESULT_PATH + path + '/' + f'Feature importance for {self.name}')

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
