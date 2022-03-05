'''
Impplementation of various regression functions


'''
import numpy as np
import pandas as pd

class MLR:   
    '''
    Multiple Linear Regression implementation with a focus on statistical inference
    '''  

    # Fit the multiple linear regression model     
    def fit(self, X, y):
        self.prednames = ['intercept'] + X.columns.to_list()
        self.n = X.shape[0]
        self.p = X.shape[1]

        # Add intercept term
        X = np.column_stack((np.ones(self.n), X.to_numpy()))
        y = y.to_numpy()

        # Calculate (X.T * X)^-1 once because used multiple times and inverse is computationally expensive
        x_t_x_inv = np.linalg.pinv(X.T @ X)
        self.b = x_t_x_inv @ X.T @ y
        self.h = X @ x_t_x_inv @ X.T
        self.e = y - X @ self.b
        
        # Breaking down sum of squared deviations and their means
        self.sst = np.sum(y ** 2) - ((np.sum(y) ** 2) / self.n)
        self.sse = self.e.T @ self.e
        self.ssr = self.sst - self.sse
        self.mse = (self.sse) / (self.n - 2)
        self.r_sq = self.ssr / self.sst

        self.s_sq_e = self.mse * (np.ones((self.n, self.n)) - self.h)
        self.cov = self.mse * x_t_x_inv

    # Get point estimate for specific values
    def predict(self, X):
        X = self.reform_x(X)
        return X @ self.b

    # Get estimated variance for specific values
    def prediction_var(self, X):
        X = self.reform_x(X)
        return X @ self.cov @ X.T

    # Helper function to format various formats of inputs (dataframe, list, etc)
    def reform_x(self, X):
        if type(X) is list:
            X = np.column_stack((np.ones(len(X)), np.reshape(X, (len(X), self.p))))
            X = X.astype('float64')
        elif isinstance(X, pd.DataFrame):
            X = np.column_stack((np.ones(X.shape[0]), X.to_numpy()))
        else:
            print('DATATYPE NOT RECOGNIZED')
        return X
