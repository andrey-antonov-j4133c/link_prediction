from sklearn.ensemble import RandomForestClassifier


class RandomForestOnFeatures:
    def __init__(self, X_train, Y_train, n_estimators=100, criterion='gini', max_depth=None, random_state=0) -> None:
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state)

        self.forest.fit(X_train, Y_train)


    def predict_proba(self, X_test):
        return self.forest.predict_proba(X_test)[:, 1]


    def accuracy_score(self, X_test, Y_test):
        return self.forest.score(X_test, Y_test)