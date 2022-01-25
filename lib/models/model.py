from tensorflow.keras import Input
from tensorflow.keras import layers as l
from tensorflow.keras.models import Model


def model(feature_names, attr_dim, attrs=True, embed_dim=32, hidden_dim=32):
    """
    A simple DL model for classification
    """

    # inputs to topological features
    feature_input = Input(shape=(len(feature_names),), name='Topological features input')

    # node_attributes inputs
    attr_input_1 = Input(shape=(attr_dim,), name='Node 1 attributes') if attr_dim > 0 else None
    attr_input_2 = Input(shape=(attr_dim,), name='Node 2 attributes') if attr_dim > 0 else None

    # dynamic representation the attributes
    c = l.Concatenate()([attr_input_1, attr_input_2])
    attrs_dyn = l.Dense(embed_dim, activation='relu', name='Dynamic_representation_of_the_attributes')(c)

    concat = l.Concatenate()([feature_input, attrs_dyn])

    if attrs:
        hidden = l.Dense(embed_dim + len(feature_names), activation='relu', name='Hidden_layer')(concat)
    else:
        hidden = l.Dense(len(feature_names), activation='relu', name='Hidden_layer')(feature_input)

    out = l.Dense(1, activation='sigmoid', name='output')(hidden)

    if attrs:
        model = Model([feature_input, attr_input_1, attr_input_2], out)
    else:
        model = Model(feature_input, out)

    model.compile(optimizer='adam', loss='poisson', metrics=['accuracy'])

    return model
