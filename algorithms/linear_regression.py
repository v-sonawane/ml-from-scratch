import numpy as np

'''
    Okay, let's take a look at the things we will need.
    
    The equation for linear regression which is y=mx+c.
    
    A function which will keep optimizing the model training.       Let's go with gradient descent. For that, we will need 3 things. Learning rate, Weights & Bias.

    So, our equation will be y=wx+b where, w=weights & b=bias.

    Note: y & x will be matrices corresponding to the datapoints in the dataset. We will keep updating the weights & bias according to the value of the differentiated form of the mean-squared error (MSE) formula multiplied by the learning rate.

'''

class LinearRegression:

    def __init__(self,lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features=X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):           

            y_pred = np.dot(X,self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

