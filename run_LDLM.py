import numpy as np
import multiprocessing 
from ldlm import LDLM
from util import *
    

def run(param):
    train_x, train_y, test_x, test_y, l1, l2, l3, rho = param
    model = LDLM(train_x, train_y, l1, l2, l3, rho)
    model.fit()
    y_pre = model.predict(test_x)
    return (str(model), (zero_one_loss(y_pre, test_y), 
                        expected_zero_one_loss(y_pre, test_y)))

    
def run_LDLM(dataset, i, train_x, train_y, test_x, test_y, scores):
    L1 = [0.001, 0.01, 0.1, 1]    
    L2 = [0.001, 0.01, 0.1, 1]
    L3 = [0.001, 0.01, 0.1, 1]
    rhos = [0.01, 0.1, 0]    
    
    params = [(train_x, train_y, test_x, test_y, l1, l2, l3, rho) 
              for l1 in L1 for l2 in L2 for l3 in L3 for rho in rhos]
    params_size = len(params)
    print("max number of models", params_size)
    
    pool = multiprocessing.Pool(40)
    finished = 0
    
    for (key, val) in pool.imap_unordered(run, params):
        finished += 1
        if finished % 200 == 0 or finished == params_size:
            print(dataset, i, params_size, finished)
        
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
        run_LDLM(dataset, i, train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "LDLM.pkl")


if __name__ == "__main__":
    datasets = ["Yeast_alpha", "Yeast_cdc", "Yeast_diau", "Yeast_elu", 
                "Yeast_heat", "Yeast_cold", "Yeast_dtt", "Yeast_spo",
                "Yeast_spo5", "Yeast_spoem", "Natural_Scene", 
                "SBU_3DFE", "SCUT_FBP", "M2B", "Human_Gene", "Movie"]
    
    datasets = ["Flickr_ldl"]
    for dataset in datasets:
        run_KF(dataset)

