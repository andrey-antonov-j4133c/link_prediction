from abc import ABC
from models.model_wrapper import ModelWrapper

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from plotting.plotting_funcs import feature_importance_gb
from settings import *


class GBModel(ModelWrapper, ABC):
    MODEL_TYPE = 'Gradient Boosting model'

    def __init__(self, name, args):
        super().__init__(name, args)

        self.n_estimators = 100
        self.criterion = 'gini'
        self.max_depth = 3
        self.random_state = args['random_state']

        self.model = self.__init_model()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict_proba(x)[:, 1]

    def feature_importance(self, path):
        importance_pd = pd.DataFrame()
        importance_pd['FI'] = pd.Series(self.model.feature_importances_, index=TOPOLOGICAL_FEATURE_NAMES)

        feature_importance_gb(
            importance_pd,
            self.name,
            RESULT_PATH + path + '/' + f'Feature importance for {self.name}')

    def plot_model(self, path):
        return

    def __init_model(self):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state)

        return model
