import numpy as np


class GaussianNaiveBayes:

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}

        for label in self.labels:
            X_label = X[y == label]
            self.means[label] = np.mean(X_label, axis=0)
            self.variances[label] = np.var(X_label, axis=0) + 1e-6
            self.priors[label] = X_label.shape[0] / X.shape[0]

    def predict(self, X):
        return np.array([self.get_posterior(x) for x in X])

    def get_likelihood(self, mean, var, x):
        coefficient = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(- (x - mean) ** 2 / (2 * var))
        return coefficient * exponent

    def get_posterior(self, x):
        posteriors = []
        for label in self.labels:
            prior = np.log(self.priors[label])
            label_conditional = np.sum(np.log(self.get_likelihood(self.means[label], self.variances[label], x)))
            posterior = prior + label_conditional
            posteriors.append(posterior)
        return self.labels[np.argmax(posteriors)]
