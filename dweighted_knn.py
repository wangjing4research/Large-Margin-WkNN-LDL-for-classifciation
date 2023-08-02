#dynamic weights, where weights admit 
#a linear dependency on distances of k nearest neighbors

import numpy as np
from util import *
from scipy.optimize import minimize

#import pymanopt
#from pymanopt.manifolds import Positive, Euclidean
#from pymanopt.optimizers import SteepestDescent


def multiply(T, W):
    return np.squeeze(T@np.expand_dims(W, axis=2))

def L1_loss(y, y_pre):
    error = y_pre - y
    sign = np.sign(error)
    return np.abs(error).sum(),  sign

def weighted_L1_loss(Y, Y_pre, W):
    error = Y_pre - Y
    sign = np.sign(error)
    loss = (W * np.abs(error)).sum()   
    return loss, W * sign

def large_margin(y_pre, L, rho):
    all_rows = np.arange(y_pre.shape[0])
    opt = y_pre[all_rows, L].reshape((-1, 1))
    hinge = np.array((opt - y_pre) <= rho, dtype = int)
    hinge[all_rows, L] = 1 - hinge.sum(1)
    loss = (hinge * y_pre).sum() - hinge[all_rows, L].sum() * rho
    
    return loss, hinge


class dweighted_knn_l1:
    '''
        x: feature matrix
        y: label distribution matrix
        k: number of nearest neighbors
        l1: balancing parameter for large margin
        rho: margin parameter
        W: re-weighting L1-norm loss
    '''
    def __init__(self, x, y, k = 11, l1 = 0.1, rho = 0.1, W = None, 
                 W_name = 0, dis = None, ind = None, l2 = 0.001):
        
        self.x = x
        self.y = y
        self.k = k
        self.l1 = l1
        self.rho = rho
        
        '''Re-weighing L1-norm loss -- refer to 
           Re-weighting Large Margin LDL for Classification
           set to None if not applied
        '''
        self.W = W
        self.W_name = W_name
        
        #initialize knns
        #dis and ind are the distances and indices of knn
        #calculate on the run if set to None
        self.__init_knns(dis, ind)
        
        #logical labels, used for large margin
        self.L = y.argmax(1)  
        
        #L2-regularization
        self.l2 = l2
        
    def __init_knns(self, dis, ind):
        if ind is None:
            dis, ind = k_nearest_neighbors(self.x, self.x, self.k + 1)
            #exclude itself
            dis = dis[:, 1:]
            ind = ind[:, 1:]

        self.dis = dis
        self.y_knn_row = self.y[ind]
        self.y_knn_col = np.transpose(self.y_knn_row, axes = [0, 2, 1])
        
        
    def objective_for_gd(self, manifold):
        
        @pymanopt.function.numpy(manifold)
        def cost(w):
            l, _ = self.objective(w.reshape(-1,))
            return l
        
        @pymanopt.function.numpy(manifold)
        def egrad(w):
            _, g = self.objective(w.reshape(-1,))
            return g.reshape((self.k, -1))
            
        return cost, egrad
    
    
    def objective(self, weights):
        w = weights.reshape(self.k, -1)@self.dis.T
        y_pre = multiply(self.y_knn_col, w.T)
        
        #regularization
        l_reg = 0.5 * self.l2 * (weights ** 2).sum()
        g_reg = self.l2 * weights
        
        
        #l1-norm loss
        if self.W is None:
            l, g = L1_loss(self.y, y_pre)
        else:
            l, g = weighted_L1_loss(self.y, y_pre, self.W)
    
        if self.l1 == 0:
            g = multiply(self.y_knn_row, g)
            g = g.T@self.dis
            return l + l_reg, g.reshape(-1,) + g_reg
        
        #large margin
        l1, g1 = large_margin(y_pre, self.L, self.rho)
        l += self.l1 * l1
        g += self.l1 * g1
        g = multiply(self.y_knn_row, g)
        g = g.T@self.dis
        
        return l + l_reg, g.reshape(-1,) + g_reg
    
    
    def solve_gd(self, max_iters = 600):
        manifold = Euclidean(self.k, self.k)
        cost, egrad = self.objective_for_gd(manifold)
        problem = pymanopt.Problem(manifold, cost, euclidean_gradient = egrad)
        solver = SteepestDescent(max_iterations = 100, verbosity = 0)
        
        self.w = solver.run(problem).point
    
    def fit(self, n_iters = 1000): 
        weights = np.ones(self.k * self.k)
        if self.k > 1:
            optimize_result = minimize(
                self.objective, weights, method = 'l-BFGS-b', jac = True,
                options = {'gtol':1e-6, 'disp': False, 'maxiter':n_iters }
            )
            self.w = optimize_result.x.reshape((self.k, -1))
        else:
            #k = 1, return 1-nn
            self.w = weights.reshape((self.k, -1))
        
    def predict(self, test_x, dis = None, ind = None):
        if ind is None:
            dis, ind = k_nearest_neighbors(self.x, test_x, self.k)
            #print(ind)
        
        w = self.w@dis.T
        #print(w)
        #pre_y = multiply(np.transpose(self.y[ind], axes = [0, 2, 1]), w.T)
        
        #clip into [0,1]
        w = np.clip(w, a_min = 1e-6, a_max = None)
        w = w / w.sum(0).reshape(1, -1)
        #print(w)
        pre_y_clip = multiply(np.transpose(self.y[ind], axes = [0, 2, 1]), w.T)
        
        #return pre_y, pre_y_clip
        return pre_y_clip
    
    
    def __str__(self):
        model_name = "dWKNNLDL_" 
        model_name += str(self.k) + "_"
        model_name += str(self.l1) + "_"
        model_name += str(self.rho) + "_"
        model_name += str(self.W_name)
        return model_name