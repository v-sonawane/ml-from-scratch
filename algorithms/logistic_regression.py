import numpy as np

def sigmoid(X):
    return 1 /(1+np.exp(-X))


class LogisticRegression():

    def __init__(self,lr=0.001,n_iters=1000):
        
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,y):

        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):

            linear_preds=np.dot(X,self.weights)+self.bias
            preds=sigmoid(linear_preds)
            
            dw=(1/n_samples) * np.dot(X.T,(preds-y))
            db=(1/n_samples) * np.sum(preds-y)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self,X):
        linear_preds=np.dot(X,self.weights)+self.bias
        preds=sigmoid(linear_preds)
        class_pred=[0 if y<=0.5 else 1 for y in preds]
        return class_pred
       

