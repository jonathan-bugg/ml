'''
Impplementation of various regression functions


'''

class MLR: 
    '''
    Multiple Linear Regression implementation with a focus on statistical inference
    '''       
    def fit(self, X, y):
        self.prednames = X.columns.to_list()
        X = np.column_stack((np.ones(X.shape[0]), X.to_numpy()))
        y = y.to_numpy()
        self.b = np.linalg.inv(X.T @ X) @ X.T @ y
        self.e = y - X @ self.b