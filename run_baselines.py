import numpy as np
from util import *
from ldl_models import SA_BFGS, AA_KNN, DAA_KNN


def run_SA_BFGS(train_x, train_y, test_x, test_y, scores):
    for c in [0.001, 0.01, 0.1, 1, 0]:
        model = SA_BFGS(train_x, train_y, c)
        model.fit()
        y_pre = model.predict(test_x)
        key = str(model)
        val = (zero_one_loss(y_pre, test_y), expected_zero_one_loss(y_pre, test_y))
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)
        

def run_AA_KNN(train_x, train_y, test_x, test_y, scores):
    for k in range(1, 51):
        aa_knn = AA_KNN()
        aa_knn.fit(train_x, train_y)
        y_pre = aa_knn.predict(test_x, k)
        key = str(aa_knn)
        val = (zero_one_loss(y_pre, test_y), expected_zero_one_loss(y_pre, test_y))
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)
        

def run_DAA_KNN(train_x, train_y, test_x, test_y, scores):
    for k in range(1, 51):
        daa_knn = DAA_KNN()
        daa_knn.fit(train_x, train_y)
        y_pre = daa_knn.predict(test_x, k)
        key = str(daa_knn)
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
        
        #run_SA_BFGS(train_x, train_y, test_x, test_y, scores)
    #save_dict(dataset, scores, "sa_bfgs.pkl")
    
        #run_AA_KNN(train_x, train_y, test_x, test_y, scores)
    #save_dict(dataset, scores, "aa_knn.pkl")
    
        run_DAA_KNN(train_x, train_y, test_x, test_y, scores)
    save_dict(dataset, scores, "daa_knn.pkl")


if __name__ == "__main__":
    datasets = ["SJAFFE", "Yeast_alpha", "Yeast_cdc", "Yeast_diau", "Yeast_elu", 
                "Yeast_heat", "Yeast_cold", "Yeast_dtt", "Yeast_spo",
                "Yeast_spo5", "Scene", "SBU_3DFE", "M2B", "SCUT_FBP", "Movie", "Flickr_ldl", "Twitter_ldl"]
    
    for dataset in datasets:
        run_KF(dataset)

