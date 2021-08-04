from sklearn.ensemble import RandomForestClassifier


class RandomForestOnFeatures:
    def __init__(self, X_train, Y_train, n_estimators=100, criterion='gini', max_depth=None, random_state=0) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state)

        self.clf.fit(X_train, Y_train)


    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)[:, 1]


    def accuracy_score(self, X_test, Y_test):
        return self.clf.score(X_test, Y_test)

    def get_model(self):
        return self.clf