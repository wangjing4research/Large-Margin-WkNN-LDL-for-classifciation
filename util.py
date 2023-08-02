from sklearn.neighbors import NearestNeighbors
import pickle, os
import numpy as np



def save_dict(dataset, scores, name):
    if not name.endswith(".pkl"):
        name += ".pkl"
    
    with open(dataset + "//" + name, 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    file_name = dataset + "//" + name
    if not os.path.exists(file_name):
        file_name += ".pkl"
    
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    

def k_nearest_neighbors(train_x, test_x, k):
    model= NearestNeighbors(n_neighbors = k, algorithm='brute', n_jobs = 1)
    model.fit(train_x)
    dis, inds = model.kneighbors(test_x)
    return dis, inds


def knn_distribution(train_y, inds, k):
    Y_hat = None
    i = 0
    for i in range(k):
        ind = inds.T[i]
        if Y_hat is None:
            Y_hat = train_y[ind]
        else:
            Y_hat += train_y[ind]
    Y_hat = Y_hat / k    
    return Y_hat



def expected_zero_one_loss(Y_pre, Y):
    Y_l = np.argmax(Y_pre, 1)
    return 1 - Y[np.arange(Y.shape[0]), Y_l].mean(0)



def zero_one_loss(Y_pre, Y):
    Y_l_pre = np.argmax(Y_pre, 1)
    Y_l = np.argmax(Y, 1)
    
    return 1 - (Y_l_pre == Y_l).mean()    