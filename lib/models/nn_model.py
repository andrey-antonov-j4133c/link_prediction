from abc import ABC

from plotting.plotting_funcs import feature_importance_keras
from settings import *
from models.model_wrapper import ModelWrapper

from tensorflow.keras import Input
from tensorflow.keras import layers as l
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


class NNModel(ModelWrapper, ABC):
    MODEL_TYPE = 'Keras NN model'

    def __init__(self, name, args):
        super().__init__(name, args)

        self.attr_dim = args['attr_dim']
        self.attributed = args['attributed']

        self.model = self.__init_model()

    def fit(self, x, y):
        self.model.fit(x, y, EPOCH)

    def predict(self, x):
        return self.model.predict(x).squeeze()

    def feature_importance(self, path):
        feature_importance_keras(
            self.model,
            attrs=self.attributed,
            path=RESULT_PATH + path + '/' + f'Feature importance for {self.name}')

    def plot_model(self, path):
        plot_model(self.model, show_shapes=True, to_file=RESULT_PATH + path + '/' + f'{self.name} model.png')

    def __init_model(self):
        # inputs to topological features
        feature_input = Input(shape=(len(TOPOLOGICAL_FEATURE_NAMES),), name='Topological features input')

        # node_attributes inputs
        if self.attributed:
            attr_input_1 = Input(shape=(self.attr_dim,), name='Node 1 attributes')
            attr_input_2 = Input(shape=(self.attr_dim,), name='Node 2 attributes')

            # dynamic representation the attributes
            c = l.Concatenate()([attr_input_1, attr_input_2])
            attrs_dyn = l.Dense(EMBED_DIM, activation='relu', name='Dynamic_representation_of_the_attributes')(c)

            concat = l.Concatenate()([feature_input, attrs_dyn])
            hidden = l.Dense(EMBED_DIM + len(TOPOLOGICAL_FEATURE_NAMES), activation='relu', name='Hidden_layer')(concat)
        else:
            hidden = l.Dense(len(TOPOLOGICAL_FEATURE_NAMES), activation='relu', name='Hidden_layer')(feature_input)

        hidden = l.Dense(32, activation='relu', name='Hidden_layer_1')(hidden)
        hidden = l.Dense(64, activation='relu', name='Hidden_layer_2')(hidden)
        hidden = l.Dense(32, activation='relu', name='Hidden_layer_3')(hidden)

        out = l.Dense(1, activation='sigmoid', name='output')(hidden)

        if self.attributed:
            model = Model([feature_input, attr_input_1, attr_input_2], out)
        else:
            model = Model(feature_input, out)

        model.compile(optimizer='adam', loss='poisson', metrics=['accuracy'])
        return model
