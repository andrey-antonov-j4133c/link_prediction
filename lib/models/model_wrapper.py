class ModelWrapper:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def feature_importance(self, path):
        raise NotImplementedError()

    def plot_model(self, path):
        raise NotImplementedError()
