import numpy as np
from SCL import LDL_SCL
from util import *

def run_LDL_SCL(train_x, train_y, test_x, test_y, scores):
    if dataset in ["Gene", "Movie", "fbp5500"]:
        l1 = 0.00001
        l2 = 0.001
        l3 = 0.0001
        c = 12
    elif dataset in ["SJAFFE", "Twitter_ldl", "Flickr_ldl"]:
        l1 = 0.0001
        l2 = 0.001
        l3 = 0.001
        c = 5
    elif dataset in ["SBU_3DFE", "M2B", "Scene", "SCUT_FBP"]:
        l1 = 0.0001
        l2 = 0.001
        l3 = 0.001
        c = 8
    else:
        l1 = 0.001
        l2 = 0.001
        l3 = 0.001
        c = 8
    
    y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c)
    key = "scl_" + str(l1) + "_" + str(l2) + "_" + str(l3) + "_" + str(c)
    val = (zero_one_loss(y_pre, test_y), expected_zero_one_loss(y_pre, test_y))
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
        run_LDL_SCL(train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "SCL.pkl")
    
    


if __name__ == "__main__":    
    
    datasets = ["Flickr_ldl", "Twitter_ldl"]
    
    for dataset in datasets:
        run_KF(dataset)

