import numpy as np
from src.linear_regression_scratch import add_bias, mse

class RidgeRegressionScratch:
    """Ridge Regression using Closed-form solution."""
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X

        n_params = X_design.shape[1]
        L = np.eye(n_params)
        if self.fit_intercept:
            L[0, 0] = 0.0  

        A = X_design.T @ X_design + self.alpha * L
        self.theta = np.linalg.pinv(A) @ X_design.T @ y
        return self

    def predict(self, X):
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X
        return X_design @ self.theta
