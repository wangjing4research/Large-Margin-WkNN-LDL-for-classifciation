#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
from scipy.special import softmax
from numpy.linalg import norm
import random
import os, pickle
import time

eps = 1e-15

def save_dict(dataset, scores, name):
    with open(dataset + "//" + name + ".pkl", 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    if not os.path.exists(dataset + "//" + name):
        name += ".pkl"
        
    with open(dataset + "//" + name, 'rb') as f:
        return pickle.load(f)



def Top_K(Y, k):
    Y_k = np.zeros_like(Y)
    k_ind = np.argsort(Y)[:, -k:]
    all_rows = np.arange(Y.shape[0])
    Y_k[all_rows.reshape((-1, 1)), k_ind] = 1 
    
    return np.asarray(Y_k, dtype=np.int)




def L2_Regularizer(weights):
    l = norm(weights, 'fro')
    return 0.5 * l, weights



"""
Loss function for LDL. 

Parameters
----------
y: array-like
The ground-truth label distribution matrix
----------

y_hat: array-like
The predicted label distribution matrix
----------

W: array-like
The weight matrix

"""
def KL(y, y_hat, W = None):
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    loss = -1 * np.sum(y * np.log(y_hat))
    grad = y_hat - y
    
    return loss, grad

def weighted_KL(y, y_hat, W):
    y = np.clip(y, 1e-15, 1)
    y_hat = np.clip(y_hat, 1e-15, 1)
    
    loss = -1 * np.sum(W * y * np.log(y_hat))
    
    y_tilde = W * y
    grad = y_hat * np.sum(y_tilde, 1).reshape((-1, 1)) - y_tilde
    
    return loss, grad


def weighted_L1(y, y_hat, W):
    diff = y_hat - y
    sign = np.sign(diff)
    
    loss = (np.absolute(diff) * W).sum()
    
    sign_y_hat = y_hat * sign * W
    grad = np.sum(-sign_y_hat, 1).reshape((-1, 1)) * y_hat + sign_y_hat
    
    return loss, grad



def L1(y, y_hat, W = None):
    diff = y_hat - y
    sign = np.sign(diff)
    
    loss = (np.absolute(diff)).sum()
    
    sign_y_hat = y_hat * sign
    grad = np.sum(-sign_y_hat, 1).reshape((-1, 1)) * y_hat + sign_y_hat
    
    return loss, grad



def L2(y, y_hat, W = None):
    diff = y_hat - y
    
    loss = norm(diff, 'fro') ** 2
    
    diff_y_hat = 2 * diff * y_hat
    grad = np.sum(-diff_y_hat, 1).reshape((-1, 1)) * y_hat + diff_y_hat
    
    return loss, grad



def weighted_L2(y, y_hat, W):
    diff = y_hat - y
    
    loss = ((diff ** 2) * W).sum()
    
    diff_y_hat = 2 * W * diff * y_hat
    grad = np.sum(-diff_y_hat, 1).reshape((-1, 1)) * y_hat + diff_y_hat
    
    return loss, grad



def jeffrey(y, y_hat, W = None):
    def weighted_ME(Y_hat, W):  
        weighted_Y_hat = Y_hat * W
        grad = np.sum(-weighted_Y_hat, 1).reshape((-1, 1)) * Y_hat + weighted_Y_hat
        return grad

    def weighted_log_ME(Y_hat, W):
        return W - W.sum(1).reshape((-1, 1)) * Y_hat
    
    l1, g1 = KL(y, y_hat)
    
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    ln_y = np.log(y)
    ln_y_hat = np.log(y_hat)
    ln_y_hat_y = ln_y_hat - ln_y
    l2 = (y_hat * ln_y_hat_y).sum()
    g2 = weighted_ME(y_hat, ln_y_hat_y) + weighted_log_ME(y_hat, y_hat)
    
    loss = l1 + l2
    grad = g1 + g2
    
    return loss, grad



def weighted_jeffrey(y, y_hat, W = None):
    def weighted_ME(Y_hat, W):  
        weighted_Y_hat = Y_hat * W
        grad = np.sum(-weighted_Y_hat, 1).reshape((-1, 1)) * Y_hat + weighted_Y_hat
        return grad

    def weighted_log_ME(Y_hat, W):
        return W - W.sum(1).reshape((-1, 1)) * Y_hat
    
    l1, g1 = KL(y, y_hat)
    
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    ln_y = np.log(y)
    ln_y_hat = np.log(y_hat)
    ln_y_hat_y = ln_y_hat - ln_y
    l2 = (W * y_hat * ln_y_hat_y).sum()
    
    g2 = weighted_ME(y_hat, W * ln_y_hat_y) + weighted_log_ME(y_hat, W * y_hat)
    
    loss = l1 + l2
    grad = g1 + g2
    
    return loss, grad


"""
The large margin function
"""
def large_margin(Y, Y_pre, neg_Y, rho):
    g = 0 
    loss = 0
    
    for i in range(Y_pre.shape[1]):
        y = Y[:, i].reshape((-1, 1))
        y_pre= Y_pre[:, i].reshape((-1, 1))
        Y_diff = (y_pre - Y_pre)
        loss_ind = np.asarray(Y_diff < rho, dtype = np.int) * y * neg_Y
        
        loss_ind[:, i] = -1 * loss_ind.sum(1)      
        loss += (-1 * loss_ind[:, i])
        g += loss_ind
    
    Y_pre_g = g * Y_pre
    grad = (np.sum(-Y_pre_g, 1).reshape((-1, 1)) * Y_pre + Y_pre_g) / (rho + eps)
    loss = loss.sum() + Y_pre_g.sum() / (rho + eps)
    
    return loss, grad



# In[8]:


def append_intercept(X):
        return np.hstack((X, np.ones(np.size(X, 0)).reshape(-1, 1)))


# In[9]:


def data_iter(batch_size, n_examples):
    idx = list(range(n_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, n_examples, batch_size)):
        index = np.array(idx[i: min(i + batch_size, n_examples)])
        yield batch_i, index


# In[10]:


def adam(params, grads, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    for param, grad, v, sqr in zip(params, grads, vs, sqrs):

        v[:] = beta1 * v + (1. - beta1) * grad
        sqr[:] = beta2 * sqr + (1. - beta2) * np.square(grad)

        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)

        div = lr * v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
        param[:] = param - div


# In[101]:


def get_min_margin(Y, Y_label):
    min_margin = 100
    for y, y_l in zip(Y, Y_label):
        y_l_not = np.asarray(1 - y_l, dtype=np.bool)
        y_l = np.asarray(y_l, dtype=np.bool)
        
        cur_min_margin = y[y_l].min() - y[y_l_not].max()
        
        if cur_min_margin <= min_margin:
            min_margin = cur_min_margin
            
    return min_margin



class RWLM_LDL:
    
    """
    The class for the Re-weighted Large Margin LDL. 
    
    Parameters
    -------------
    X: feature matrix
    Y: label distribution matrix
    Y_label: the label matrix
    W: the weight matrix
    loss: {KL, weight_KL, L1, weighted_L1}
    C1, C2: the balance parameters
    rho: if rho is None, then the minmal margin is used
    """
    def __init__(self, X, Y, Y_label, W = None, loss = weighted_L1, C1 = 1, C2 = 1, rho = None, C = 0):
        
        
        self.X = append_intercept(X)
        self.Y = Y
        
        if W is None:
            self.W = np.ones_like(Y)
        else:
            self.W = W
        
        self.C = C
        self.C1 = C1
        self.C2 = C2
        
        self.loss = loss
        
        self.Y_label = np.asarray(Y_label, dtype = np.int)
        self.neg_Y_label = np.asarray(1 - Y_label, dtype=np.int)
        
        self.n_examples, self.n_outputs = Y.shape
        self.n_features = self.X.shape[1]
        
        self.rho = rho
        if rho is None:
            self.rho_0 = get_min_margin(Y, Y_label)
        else:
            self.rho_0 = rho
    
    
    """
        batch gradient
        
        Parameters
        ----------
        X: batch feature
        Y: batch label distribution 
        Y_label: batch label 
    """
    
    def output(self, weights, X = None, Y = None, Y_label = None, neg_Y_label = None, W = None):      
        if X is None:
            X = self.X
            Y = self.Y
            Y_label = self.Y_label
            neg_Y_label = self.neg_Y_label
            W = self.W
        
        batch_size = np.size(X, 0)
        Y_hat = softmax(np.dot(X, weights), axis = 1)

        #regularization
        l2_reg, l2_reg_grad = L2_Regularizer(weights)
        
        #loss function: weighted_L1
        if self.C1 == 0:
            func_loss, func_grad = 0, 0
        else:
            func_loss, func_grad = self.loss(Y, Y_hat, W)   

        
        #large margin 
        if self.C2 == 0:
            margin_loss, margin_grad = 0, 0
        else:
            margin_loss, margin_grad = large_margin(Y_label, Y_hat, neg_Y_label, self.rho_0)

        loss = self.C * l2_reg + (self.C1 * func_loss + self.C2 * margin_loss) #/ batch_size
        grad = self.C * l2_reg_grad + (np.dot(X.T, (self.C1 * func_grad + self.C2 * margin_grad))) #/ batch_size
        
        return loss, grad
    
    def solve_adam(self, batch_size = 32, lr = 0.001, epochs = 200, period = 1000, verbose = False):
        if self.C1 == 0 and self.C2 == 0:
            self.weights = np.zeros((self.n_features, self.n_outputs))
            return
        
        weights = np.random.normal(0, 1, (self.n_features, self.n_outputs))
        
        vs = []
        sqrs = []
        vs.append(np.zeros_like(weights))
        sqrs.append(np.zeros_like(weights))
        
        l, _ = self.output(weights)
        total_loss = [l]
        
        t = 0
        for epoch in range(1, epochs + 1):
            #t1 = time.time()
            for batch_i, index in data_iter(batch_size, self.n_examples):
                batch_X, batch_Y = self.X[index], self.Y[index]
                batch_Y_label, batch_neg_Y_label = self.Y_label[index], self.neg_Y_label[index]
                batch_W = self.W[index]
                
                _, grad = self.output(weights, batch_X, batch_Y, batch_Y_label, batch_neg_Y_label, batch_W)
        
                t += 1
                adam([weights], [grad], vs, sqrs, lr, batch_size, t)
            #print(time.time() - t1)
            if verbose:
                if epoch % 100 == 0:
                    l, _ = self.output(weights)
                    total_loss.append(l)
                    print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" % 
                              (batch_size, lr, epoch, total_loss[-1]))
            
            
        self.weights = weights
        
        if verbose:
            return total_loss
        
        
    def predict(self, test_X):
        test_X = append_intercept(test_X)
        test_Y_hat = softmax(np.dot(test_X, self.weights), axis = 1)  
        return test_Y_hat
    
    def return_model(self):
        return self.weights
        
    def __str__(self):
        model_str = "LDL4C_"
        model_str += str(self.C) + "_"
        model_str += str(self.C1) + "_"
        model_str += str(self.C2) + "_"
        
        if self.loss is L1:
            model_str += "L1_"
        elif self.loss is weighted_L1:
            model_str += "WL1_"
        elif self.loss is KL:
            model_str += "KL_"
        elif self.loss is weighted_KL:
            model_str += "WKL_"
        elif self.loss is L2:
            model_str += "L2_"
        elif self.loss is weighted_L2:
            model_str += "WL2_"
        elif self.loss is jeffrey:
            model_str += "J_"
        elif self.loss is weighted_jeffrey:
            model_str += "WJ_"
        
        if self.rho is None:
            model_str += "min"
        else:
            model_str += str(self.rho)
            
        return model_str

