import numpy as np
from util import *
from scipy.optimize import minimize
from ldl_metrics import score

#import pymanopt
#from pymanopt.manifolds import Positive, Euclidean
#from pymanopt.optimizers import SteepestDescent

def multiply(T, W):
    return np.squeeze(T@np.expand_dims(W, axis=2)).sum(0)

def large_margin(Y_pre, L, rho):
    all_rows = np.arange(Y_pre.shape[0])
    opt = Y_pre[all_rows, L].reshape((-1, 1))
    
    hinge = np.array((opt - Y_pre) < rho, dtype = int)
    hinge[all_rows, L] = 1 - hinge.sum(1)
    
    loss = (hinge * Y_pre).sum() - hinge[all_rows, L].sum() * rho
    
    return loss, hinge


def L1_loss(Y, Y_pre):
    error = Y_pre - Y
    sign = np.sign(error)
    
    loss = np.abs(error).sum()  
    
    return loss, sign


def weighted_L1_loss(Y, Y_pre, W):
    error = Y_pre - Y
    sign = np.sign(error)
    
    loss = (W * np.abs(error)).sum()   
    
    return loss, W * sign


class weighted_knn_l1:
    def __init__(self, x, y, k = 11, l1 = 1, rho = 0.1, W = None, W_i = 0, inds = None):
        self.x = x
        self.y = y
        self.k = k
        self.l1 = l1
        
        #re-weighted w.r.t. label description degrees
        self.W = W 
        self.W_i = W_i
        
        #initialize knns
        self.__init_knns(inds)
        
        #logical labels
        self.L = y.argmax(1)
        
        #rho for large margin
        self.rho = rho
        
    def __init_knns(self, inds):
        if inds is None:
            _, inds = k_nearest_neighbors(self.x, self.x, self.k + 1)
            inds = inds[:, 1:]
        self.y_knn_row = self.y[inds]
        self.y_knn_col = np.transpose(self.y_knn_row, axes = [0, 2, 1])    
        
    def objective(self, w):
        y_pre = self.y_knn_col@w
        
        #L1-norm loss and gradient
        if self.W is None:
            l, g = L1_loss(self.y, y_pre)
        else:
            l, g = weighted_L1_loss(self.y, y_pre, self.W)
            
        if self.l1 == 0:
            grad = multiply(self.y_knn_row, g)
            return l, grad
        
        #Large-margin
        l1, g1 = large_margin(y_pre, self.L, self.rho)
        
        loss = l + self.l1 * l1
        grad = multiply(self.y_knn_row, g + self.l1 * g1)
        
        return loss, grad
    
    
    def objective_for_gd(self, manifold):
        
        @pymanopt.function.numpy(manifold)
        def cost(w):
            l, _ = self.objective(w)
            return l
        
        @pymanopt.function.numpy(manifold)
        def egrad(w):
            _, g = self.objective(w)
            return g
        
        return cost, egrad


    def solve_gd(self, max_iters = 600):
        manifold = Euclidean(self.k)
        cost, egrad = self.objective_for_gd(manifold)
        problem = pymanopt.Problem(manifold, cost, euclidean_gradient = egrad)
        solver = SteepestDescent(verbosity = 0, max_iterations = 100)
        
        self.w = solver.run(problem).point.reshape(-1, )
    
    
    def fit(self, n_iters = 1000):  
        weights = np.ones((self.k,)) / self.k    
        if self.k > 1:#k == 1, return the 1-nn
            optimize_result = minimize(self.objective, weights.reshape(-1,), method = 'l-BFGS-b', jac = True,
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':n_iters })
            self.w = optimize_result.x
        else:
            self.w = weights
    
        
    def predict(self, test_x, inds = None):
        if inds is None:
            _, inds = k_nearest_neighbors(self.x, test_x, self.k)
        #pre_y = np.transpose(self.y[inds], axes = [0, 2, 1])@self.w
        
        #clip w into [0,1]
        self.w = np.clip(self.w, a_min = 1e-5, a_max = None)
        self.w = self.w / self.w.sum()
        pre_y_clip = np.transpose(self.y[inds], axes = [0, 2, 1])@self.w
        #return pre_y, pre_y_clip
        return pre_y_clip
    
    
    def __str__(self):
        model_name = "WKNNLDL_" 
        model_name += str(self.k) + "_"
        model_name += str(self.l1) + "_"
        model_name += str(self.rho) +"_"
        model_name += str(self.W_i)
        return model_name