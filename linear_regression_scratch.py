import numpy as np

def add_bias(X):
    """Add bias (intercept) column."""
    X = np.asarray(X)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def mse(y_true, y_pred):
    """Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred)**2)

class LinearRegressionScratch:
    """Closed-form Linear Regression with optional Ridge regularization."""
    def __init__(self, fit_intercept=True, ridge_lambda=0.0):
        self.fit_intercept = fit_intercept
        self.ridge_lambda = float(ridge_lambda)
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X

        n_params = X_design.shape[1]
        if self.ridge_lambda == 0.0:
            self.theta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        else:
            L = np.eye(n_params)
            if self.fit_intercept:
                L[0,0] = 0.0
            A = X_design.T @ X_design + self.ridge_lambda * L
            self.theta = np.linalg.pinv(A) @ X_design.T @ y
        return self

    def predict(self, X):
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X
        return X_design @ self.theta


class LinearRegressionGD:
    """Gradient Descent Linear Regression with optional Ridge regularization."""
    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True, ridge_lambda=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.ridge_lambda = ridge_lambda
        self.theta = None
        self.verbose = verbose
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1,)
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X

        n, p = X_design.shape
        self.theta = np.zeros(p)
        for it in range(self.n_iter):
            preds = X_design @ self.theta
            error = preds - y
            grad = (2.0 / n) * (X_design.T @ error)

            if self.ridge_lambda != 0.0:
                reg = 2.0 * self.ridge_lambda * self.theta
                if self.fit_intercept:
                    reg[0] = 0.0
                grad += reg

            self.theta -= self.lr * grad
            loss = np.mean(error**2) + self.ridge_lambda * np.sum(self.theta**2)
            self.loss_history.append(loss)

            if self.verbose and it % (self.n_iter // 5 + 1) == 0:
                print(f"Iteration {it}, Loss = {loss:.4f}")
        return self

    def predict(self, X):
        if self.fit_intercept:
            X_design = add_bias(X)
        else:
            X_design = X
        return X_design @ self.theta
