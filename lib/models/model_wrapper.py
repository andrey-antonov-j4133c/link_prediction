class ModelWrapper:
    def __init__(self, feature_cols, name, args, type='full'):
        self.feature_cols = feature_cols
        self.name = name
        self.args = args

    def fit(self, node_df, y_col):
        raise NotImplementedError()

    def predict(self, node_df):
        raise NotImplementedError()

    def feature_importance(self, train_samples, test_samples, path):
        raise NotImplementedError()

    def plot_model(self, path):
        raise NotImplementedError()
