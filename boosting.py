from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            blend: bool = False,
            base_model_class = DecisionTreeRegressor
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

        self.meta_X = None
        self.blend = blend
        self.meta_model = LinearRegression() if self.blend else None

    def fit_new_base_model(self, x, y, predictions):
        num_samples_to_draw = int(self.subsample * x.shape[0])
        subsample_indices = np.random.choice(np.arange(x.shape[0]), num_samples_to_draw)
        selected_features = x[subsample_indices, :]
        selected_labels = y[subsample_indices]
        target_values = -self.loss_derivative(selected_labels, predictions[subsample_indices])

        fitted_model = None
        if self.base_model_class is DecisionTreeRegressor:
            tree_model = self.base_model_class(**self.base_model_params)
            tree_model.fit(selected_features, target_values)
            optimal_gamma = self.find_optimal_gamma(y, predictions, tree_model.predict(x))
            adjusted_gamma = self.learning_rate * optimal_gamma
            self.gammas.append(adjusted_gamma)
            fitted_model = tree_model
        elif self.base_model_class is LogisticRegression:
            logistic_model = self.base_model_class().fit(selected_features, selected_labels)
            fitted_model = logistic_model

        self.models.append(fitted_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """

        if self.blend:
            self.early_stopping_rounds = None
            self.meta_X = np.empty((y_valid.shape[0], self.n_estimators))

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for estimator_index in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions = self.models[-1].predict(x_train)
            valid_predictions = self.models[-1].predict(x_valid)

            if self.blend:
                self.meta_X[:, estimator_index] = valid_predictions

            if self.plot:
                train_score = self.score(x_train, y_train)
                valid_score = self.score(x_valid, y_valid)

                self.history['train'].append(train_score)
                self.history['valid'].append(valid_score)

            if self.early_stopping_rounds is not None:
                self.validation_loss[estimator_index] = self.loss_fn(y_valid, valid_predictions)
                if (estimator_index + 1 >= self.early_stopping_rounds and
                        self.validation_loss[estimator_index] <= self.validation_loss[estimator_index - 1]):
                    break

        if self.blend:
            self.meta_model.fit(self.meta_X, y_valid)

        if self.plot:
            sns.lineplot(data=self.history)
            plt.title('Performance Over Iterations')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Score')
            plt.legend(['Train Score', 'Validation Score'])
            plt.show()

    def predict_proba(self, x):
        if self.blend:
            meta_features = np.empty((x.shape[0], self.n_estimators))

            for estimator_index, model in enumerate(self.models):
                if self.base_model_class is DecisionTreeRegressor:
                    meta_features[:, estimator_index] = model.predict(x)
                elif self.base_model_class is LogisticRegression:
                    meta_features[:, estimator_index] = model.predict_proba(x)[:, 1]

            probabilities = np.zeros((x.shape[0], 2))

            if self.base_model_class is DecisionTreeRegressor:
                probabilities[:, 1] = self.sigmoid(self.meta_model.predict(meta_features))
            elif self.base_model_class is LogisticRegression:
                probabilities[:, 1] = self.meta_model.predict(meta_features)

            probabilities[:, 0] = 1 - probabilities[:, 1]

        else:
            probabilities = np.zeros((x.shape[0], 2))

            for gamma, model in zip(self.gammas, self.models):
                probabilities[:, 1] += gamma * model.predict(x)

            probabilities[:, 1] = self.sigmoid(probabilities[:, 1])

            probabilities[:, 0] = 1 - probabilities[:, 1]

        return probabilities

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        f_imps = np.mean(np.array([model.feature_importances_ for model in self.models]), axis=0)
        return f_imps
