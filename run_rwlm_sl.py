import numpy as np
from rwlm_ldl import *
import scipy.stats as ss
import multiprocessing 
from util import *

    
def Top_K(Y, k):
    Y_k = np.zeros_like(Y)
    k_ind = np.argsort(Y)[:, -k:]
    all_rows = np.arange(Y.shape[0])
    Y_k[all_rows.reshape((-1, 1)), k_ind] = 1 
    
    return np.asarray(Y_k, dtype=np.int)
    
    

def run(para):
    train_x, train_y, train_y_label, test_x, test_y, W, l, c1, c2, rho, c = para
    
    model = RWLM_LDL(train_x, train_y, train_y_label, W, l, c1, c2, rho, c)
    model.solve_adam(verbose=True, batch_size=min(256, train_x.shape[0]), lr = 0.001, epochs=800)
    y_pre = model.predict(test_x)
    
    return (str(model), (zero_one_loss(y_pre, test_y), expected_zero_one_loss(y_pre, test_y)))
    


def run_RWLMLDL(dataset, train_x, train_y, test_x, test_y, scores):
    
    C_0 = [0.0001]
    C_1 = [0.001, 0.01, 0.1, 0, 1, 10, 100, 1000]    
    C_2 = [0.001, 0.01, 0.1, 0, 1, 10, 100, 1000]
    rhos = [0.001, 0.01, 0.1, 0]
    
    
    W = ss.entropy(train_y.T).reshape((-1, 1)) * train_y    
    
    train_y_label = Top_K(train_y, 1)
    
    params = [(train_x, train_y, train_y_label, test_x, test_y, W, weighted_L1, c1, c2, rho, c) 
                                        for c1 in C_1 for c2 in C_2 for rho in rhos for c in C_0]
    params_size = len(params)
    print("max number of models", params_size)
    
    pool = multiprocessing.Pool(40)
    finished = 0
    #imap_unordered
    for (key, val) in pool.imap_unordered(run, params):
        finished += 1
        if finished % 100 == 0 or finished == params_size:
            print(dataset, params_size, finished)
        
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)
    
    pool.close()
    pool.join()

    

def run_KF(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    
        
    train_inds = load_dict(dataset, "train_fold")
    test_inds = load_dict(dataset, "test_fold")
    folds = 10
    scores = dict()

    for i in range(folds):
        print("fold", i + 1)
        
        train_x, train_y = X[train_inds[i + 1]], Y[train_inds[i + 1]]
        test_x, test_y = X[test_inds[i + 1]], Y[test_inds[i + 1]]
        
        run_RWLMLDL(dataset, train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "rwlm_ldl_mc")


if __name__ == '__main__':
    
    datasets = ["Flickr_ldl"]
    
    
    for data_set in datasets:
        run_KF(data_set)

