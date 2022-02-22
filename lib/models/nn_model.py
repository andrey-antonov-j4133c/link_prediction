import time
from abc import ABC

import numpy as np
import pandas as pd

from plotting.plotting_funcs import feature_importance
from settings import *
from models.model_wrapper import ModelWrapper

from tensorflow.keras import Input
from tensorflow.keras import layers as l
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, Sequence

import shap


class NNModel(ModelWrapper, ABC):
    MODEL_TYPE = 'Keras NN model'

    def __init__(self, feature_cols, name, args, type='full'):
        super().__init__(feature_cols, name, args, type)

        self.batch_size = 1024
        self.attr_dim = args['attr_dim']
        self.attributed = args['attributed']

        if type not in ('full', 'selected_features'):
            raise ValueError(f'"type" argument supposed to have ether "full" or "selected_features"\nGot {type}')

        self.type = type

        if self.type == 'full':
            self.model = self.__init_full_model()
        else:
            self.model = self.__init_single_input_model()

    def fit(self, node_df, y_col):
        X, y = self.__get_data(node_df, y_col)
        self.model.fit(X, y, epochs=EPOCH, verbose=1)

    def predict(self, node_df, path=None):
        X, _ = self.__get_data(node_df)
        stat = time.time()
        res = self.model.predict(X, verbose=1).squeeze()
        end = time.time()
        if path:
            pd.DataFrame({"Predict time": [end-stat]}).to_csv(RESULT_PATH + path + '/' + f'predict_time_for_{self.name}.csv')
        return res

    def feature_importance(self, train_samples, test_samples, path):
        X_train, _ = self.__get_data(train_samples)
        X_test, _ = self.__get_data(test_samples)

        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        explainer = shap.DeepExplainer(self.model, X_train)
        shap_values = explainer.shap_values(X_test)

        importance_pd = pd.DataFrame(
            shap_values[0],
            columns=self.__get_cols())

        top_important_features = importance_pd.mean(axis=0).sort_values(ascending=False)
        top_important_features = top_important_features\
            .reindex(top_important_features.map(lambda x: x).abs().sort_values(ascending=False).index)

        feature_importance(
            top_important_features,
            self.name,
            path=RESULT_PATH + path + '/' + f'Feature importance for {self.name}.pdf')

        return top_important_features

    def plot_model(self, path, name='model'):
        plot_model(self.model, show_shapes=True, to_file=RESULT_PATH + path + '/' + f'{self.name} {name}.pdf')

    def __init_full_model(self):
        input_length = len(self.feature_cols) + (self.attr_dim*2 if self.attributed else 0)
        input = Input(shape=(input_length,), name='Input')

        if self.attributed:
            features = l.Lambda(lambda x: x[:, :len(self.feature_cols)])(input)
            node_1_attrs = l.Lambda(lambda x: x[:, len(self.feature_cols):len(self.feature_cols)+self.attr_dim])(input)
            node_2_attrs = l.Lambda(lambda x: x[:, len(self.feature_cols) + self.attr_dim:])(input)

            node_1_embed = l.Dense(EMBED_DIM, activation='relu', name='Node_one_embed')(node_1_attrs)
            node_2_embed = l.Dense(EMBED_DIM, activation='relu', name='Node_two_embed')(node_2_attrs)

            concat = l.Concatenate()([features, node_1_embed, node_2_embed])
            hidden = l.Dense(32, activation='relu', name='Hidden_layer')(concat)
        else:
            hidden = l.Dense(32, activation='relu', name='Hidden_layer')(input)

        hidden = l.Dense(64, activation='relu', name='Hidden_layer_two')(hidden)
        hidden = l.Dense(64, activation='relu', name='Hidden_layer_three')(hidden)
        hidden = l.Dense(32, activation='relu', name='Hidden_layer_four')(hidden)

        out = l.Dense(1, activation='sigmoid', name='output')(hidden)

        model = Model(input, out)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def __init_single_input_model(self):
        input_length = len(self.__get_cols())
        input = Input(shape=(input_length,), name='Input')

        hidden = l.Dense(32, activation='relu', name='Hidden_layer')(input)
        hidden = l.Dense(64, activation='relu', name='Hidden_layer_two')(hidden)
        hidden = l.Dense(64, activation='relu', name='Hidden_layer_three')(hidden)
        hidden = l.Dense(32, activation='relu', name='Hidden_layer_four')(hidden)

        out = l.Dense(1, activation='sigmoid', name='output')(hidden)

        model = Model(input, out)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def __get_data(self, node_df, y_col=None):
        X = node_df[self.__get_cols()].values
        y = node_df[y_col].values if y_col else None
        return X, y

    def __get_cols(self):
        if self.type != 'full':
            return self.feature_cols
        if self.attributed:
            return self.feature_cols \
                + [f'node_1_attr_{i}' for i in range(self.attr_dim)] \
                + [f'node_2_attr_{i}' for i in range(self.attr_dim)]
        else:
            return self.feature_cols


class CustomDataGen(Sequence):
    def __init__(
            self,
            node_df,
            feature_cols,
            y_col,
            batch_size,
            shuffle=True):

        self.node_df = node_df.copy()

        self.feature_cols = feature_cols
        self.y_col = y_col

        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle:
            self.node_df = self.node_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        node_batch = self.node_df.iloc[index:index + self.batch_size]

        if self.y_col:
            return (node_batch[self.feature_cols].values,
                    np.array(node_batch['node1_attrs'].values.tolist()),
                    np.array(node_batch['node2_attrs'].values.tolist())), node_batch[self.y_col].values
        else:
            return (node_batch[self.feature_cols].values,
                    np.array(node_batch['node1_attrs'].values.tolist()),
                    np.array(node_batch['node2_attrs'].values.tolist())), None

    def __len__(self):
        n = int(len(self.node_df) / self.batch_size)
        if n * self.batch_size < len(self.node_df):
            n += 1
        return n
