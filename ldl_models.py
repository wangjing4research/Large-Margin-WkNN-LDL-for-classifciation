import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax

eps = np.finfo(np.float32).eps

class SA_BFGS:
    def __init__(self, x, y, C = 0):
        self.C = C
        self.x = x
        self.y = y
        self.n_features, self.n_outputs = self.x.shape[1], self.y.shape[1]

    def predict(self, x):
        p_yx = x@self.W
        p_yx = softmax(p_yx, axis = 1)
        return p_yx

    def object_fun(self, weights):
        W = weights.reshape(self.n_outputs, self.n_features).transpose()
        p_yx = self.x@W
        y_pre = softmax(p_yx, axis = 1)
        
        func_loss = self.loss(y_pre) + self.C * 0.5 * (weights ** 2).sum() 
        func_grad = self.gradient(y_pre) + self.C * weights
        
        return func_loss, func_grad

    def gradient(self, y_pre):
        grad = self.x.T@(y_pre - self.y)
        return grad.transpose().reshape(-1, ) 

    def loss(self, y_pre):
        y_true = np.clip(self.y, 1e-7, 1)
        y_pre = np.clip(y_pre, 1e-7, 1)
        return -1 * np.sum(y_true * np.log(y_pre))

    def fit(self, n_iters = 500):        
        weights = np.random.uniform(-0.1, 0.1, self.n_features * self.n_outputs)
        optimize_result = minimize(self.object_fun, weights, method = 'l-BFGS-b', jac = True,
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':n_iters })
        
        weights = optimize_result.x
        self.W = weights.reshape(self.n_outputs, self.n_features).transpose()
        
    def __str__(self):
        name = "SA-BFGS_" +  str(self.C)
        return name
    

class AA_KNN:
    def __init__(self):
        self.model =NearestNeighbors(n_neighbors = 50, algorithm='brute', n_jobs = -1)
    
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.model.fit(self.train_x)
    
    def predict(self, test_x, k):
        self.k = k
        _, inds = self.model.kneighbors(test_x)
        Y_hat = None
        i = 0
        for i in range(k):
            ind = inds.T[i]
            if Y_hat is None:
                Y_hat = self.train_y[ind]
            else:
                Y_hat += self.train_y[ind]
        Y_hat = Y_hat / k    
        
        return Y_hat
    
    def __str__(self):
        return "AA-KNN_" + str(self.k)
    
    

def weights(dis):
    dis = np.clip(dis, a_min = eps, a_max=None)
    w = np.clip(np.reshape(dis[:,-1], (-1, 1)) - dis, a_min=eps, a_max=None) / np.clip(np.reshape(dis[:, -1] - dis[:, 0], (-1, 1)), a_min=eps, a_max=None)
    
    w = w / w.sum(1).reshape((-1, 1))
    return w
    

class DAA_KNN:
    def __init__(self):
        self.model =NearestNeighbors(algorithm='brute', n_jobs = -1)
    
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.model.fit(self.train_x)
    
    def predict(self, test_x, k):
        self.k = k
        dis, inds = self.model.kneighbors(test_x, n_neighbors = k)
        w = weights(dis)
        
        Y_hat = None
        i = 0
        for i in range(k):
            ind = inds.T[i]
            if Y_hat is None:
                Y_hat = (self.train_y[ind] * np.reshape(w[:, i], (-1, 1)))
            else:
                Y_hat += (self.train_y[ind] * np.reshape(w[:, i], (-1, 1)))
        
        return Y_hat
    
    def __str__(self):
        return "DAA-KNN_" + str(self.k)