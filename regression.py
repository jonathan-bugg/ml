'''
Impplementation of various regression functions

'''
import numpy as np
import pandas as pd
from pyparsing import alphanums
import scipy

class MLR:   
    '''
    Implementation of multiple linear regression with a focus on statistical inference
    '''
    # Fit the multiple linear regression model     
    def fit(self, X, y):
        self.prednames = ['intercept'] + X.columns.to_list()
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.df = self.n - self.p
        X = np.column_stack((np.ones(self.n), X.to_numpy()))
        y = y.to_numpy()

        x_t_x_inv = np.linalg.pinv(X.T @ X)
        self.b = x_t_x_inv @ X.T @ y
        self.h = X @ x_t_x_inv @ X.T
        self.e = y - X @ self.b
        
        # Breaking down sum of squared deviations and their means
        self.sst = np.sum(y ** 2) - ((np.sum(y) ** 2) / self.n)
        self.sse = self.e.T @ self.e
        self.ssr = self.sst - self.sse
        self.mse = (self.sse) / (self.n - self.p + 1)
        self.msr = self.ssr / self.p
        self.r_sq = self.ssr / self.sst

        self.s_sq_e = self.mse * (np.ones((self.n, self.n)) - self.h)
        self.cov = self.mse * x_t_x_inv

    # Get point estimate for specific values
    def predict(self, X, include_ci = False, a = 0.05, mean_or_pred = 'pred'):
        X = self.reform_x(X)
        pred = X @ self.b
        if include_ci:
            # CI for predicing mean response has different CI than predicitng new observation
            t_val = scipy.stats.t.ppf(1 - (a/2), self.n - self.p)
            if mean_or_pred == 'mean':
                s_sq_y_hat = X @ self.cov @ X.T
                lb = pred - t_val * np.sqrt(s_sq_y_hat)
                ub = pred + t_val * np.sqrt(s_sq_y_hat)
            # Defining CI bounds if predicting new observation
            elif mean_or_pred == 'pred':
                s_sq_pred = self.mse * (1 + X @ self.cov @ X.T)
                lb = pred - t_val * np.sqrt(s_sq_pred)
                ub = pred + t_val * np.sqrt(s_sq_pred)
            return pred, [lb, ub]
        else:
            return pred

    # Get estimated variance for specific values
    def prediction_var(self, X):
        X = self.reform_x(X)
        return X @ self.cov @ X.T

    # Hypothesis test of whether B_1 = ... = B_p = 0
    def test_reg_rel(self, a):
        test_val = self.msr / self.mse
        f_val = scipy.stats.f.ppf(a, self.p, self.n - self.p)
        if f_val > test_val:
            concl = 'accept null'
        else:
            concl = 'reject null'

        return test_val, f_val, concl

    def reform_x(self, X):
        if type(X) is list:
            if type(X[0]) is list:
                X = np.column_stack((np.ones(len(X)), np.reshape(X, (len(X), self.p))))
                X = X.astype('float64')
            else:
                X = np.column_stack((1, np.reshape(X, (1, self.p))))
                X = X.astype('float64')
        elif isinstance(X, pd.DataFrame):
            X = np.column_stack((np.ones(X.shape[0]), X.to_numpy()))
        
        else:
            print('DATATYPE NOT RECOGNIZED')
        return X

def two_mod_f_test(reg_red, reg_full, a = 0.05):
    test_stat = ((reg_red.sse - reg_full.sse) / (reg_red.df - reg_full.df)) / (reg_full.sse / reg_full.df)
    f_val = scipy.stats.f.ppf(1 - a, reg_full.p - reg_red.p, reg_full.n - reg_full.p)
    if f_val >= test_stat:
        conclusion = 'accept null'
    else:
        conclusion = 'reject null'
    return conclusion, test_stat, f_val


# class shrinkReg:

#     def __init__(self, lam, alpha, learn_rate):
#         self.lam = lam
#         self.alpha = alpha
#         self.learn_rate = learn_rate


#     def fit(self, X, y):
        