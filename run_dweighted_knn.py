import os
import numpy as np
import scipy.stats as ss
from util import *
from dweighted_knn import dweighted_knn_l1


def run(param):
    train_x, train_y, test_x, test_y, k, l1, rho, \
            W, W_name, dis0, ind0, dis1, ind1 = param 
    model = dweighted_knn_l1(train_x, train_y, k, l1, 
                             rho, W, W_name, dis0, ind0)
    model.fit(1000)
    y_pre, _ = model.predict(test_x, dis1, ind1)
    return (str(model), 
            (zero_one_loss(y_pre, test_y), 
                expected_zero_one_loss(y_pre, test_y))
           )


#调参
def run_dweighted_knn(dataset, train_x, train_y, test_x, test_y, scores):
    #parameters
    L1 = [0.001, 0.01, 0.1, 1]
    Rho = [0.1]

    Ws = [ss.entropy(train_y.T).reshape(-1, 1) * train_y]
    W_names = [0]

    Ks = [11]
    
    params = []
    dis0, ind0 = k_nearest_neighbors(train_x, train_x, 12)
    dis1, ind1 = k_nearest_neighbors(train_x, test_x, 12)
    
    for k in Ks:
        param = [(train_x, train_y, test_x, test_y, k, l1, rho, Ws[i], i, 
                  dis0[:, 1:k+1], ind0[:, 1:k+1], dis1[:, :k], ind1[:, :k])
                 for l1 in L1 for rho in Rho for i in W_names] 
        params.extend(param)
    
    for (key, val) in map(run, params):
        
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)

    

def run_KF(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_fold")
    test_inds = load_dict(dataset, "test_fold")
    scores = dict()

    for i in range(10):
        print("fold", i + 1)

        train_x, train_y = X[train_inds[i + 1]], Y[train_inds[i + 1]]
        test_x, test_y = X[test_inds[i + 1]], Y[test_inds[i + 1]]
        run_dweighted_knn(dataset, train_x, train_y, test_x, test_y, scores)
        
        save_dict(dataset, scores, "dweighted_knn_" + str(i)+ ".pkl")
        

if __name__ == "__main__":
    datasets = ["SJAFFE"]
    
    for dataset in datasets:
        run_KF(dataset)

