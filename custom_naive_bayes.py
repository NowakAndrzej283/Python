import numpy as np


class CustomGaussianNB:


    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

        self.epsilon = 1e-9

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)


        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + self.epsilon

            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):


        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):

        posteriors = []

        # Obliczanie prawdopodobieństwa
        for idx, c in enumerate(self.classes):
            # Logarytm prawdopodobieństwa a priori
            prior_log = np.log(self.priors[idx])

            # Logarytm gęstości prawdopodobieństwa (likelihood)
            # Używamy logarytmów, aby uniknąć problemów z bardzo małymi liczbami (underflow)
            likelihood_log = np.sum(np.log(self._pdf(idx, x)))

            # Logarytm prawdopodobieństwa a posteriori
            posterior_log = prior_log + likelihood_log
            posteriors.append(posterior_log)


        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):

        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator