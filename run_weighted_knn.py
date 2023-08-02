import os
import numpy as np
import scipy.stats as ss
from util import *
from weighted_knn import weighted_knn_l1

def run(param):
    train_x, train_y, test_x, test_y, k, l1, rho, W, W_i, ind0, ind1 = param 
    model = weighted_knn_l1(train_x, train_y, k, l1, rho, W, W_i, ind0)
    model.fit(1000)
    y_pre, _ = model.predict(test_x, ind1)
    return (str(model), (zero_one_loss(y_pre, test_y), 
                        expected_zero_one_loss(y_pre, test_y)))


#调参
def run_weighted_knn(dataset, train_x, train_y, test_x, test_y, scores):
    
    #all tunable parameters
    L1 = [0.001, 0.01, 0.1, 1]
    Rho = [0.1]

    #加权的L1-norm loss
    Ws = [ss.entropy(train_y.T).reshape(-1, 1) * train_y]
    W_names = [0]  

    Ks = [11]

    params = []
    _, ind0 = k_nearest_neighbors(train_x, train_x, 12)
    _, ind1 = k_nearest_neighbors(train_x, test_x, 12)

    for k in Ks:
        param = [(train_x, train_y, test_x, test_y, k, l1, 
                  rho, Ws[W_i], W_i, ind0[:, 1:k+1], ind1[:, :k]) 
                 for l1 in L1 for rho in Rho for W_i in W_names] 
        params.extend(param)

    for (key, val) in map(run, params)
        
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
        run_weighted_knn(dataset, train_x, train_y, test_x, test_y, scores)
        save_dict(dataset, scores, "weighted_knn_" + str(i) + ".pkl")


if __name__ == "__main__":
    datasets = ["SJAFFE"]
    
    for dataset in datasets:
        run_KF(dataset)

