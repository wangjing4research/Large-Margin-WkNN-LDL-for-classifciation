# Large Margin Weighted k-Nearest Neighbors Label Distribution Learning for Classification

Code for our TNNLS'23 paper "Large Margin Weighted k-Nearest Neighbors Label Distribution Learning for Classification"

In this article, we design two novel LDL methods based on the k-Nearest Neighbors (kNN) approach for classification. First, we propose the Large margin Weighted kNN LDL (LW-kNNLDL). It learns a weight vector for the kNN algorithm to learn label distribution and implement large margin to address the objective inconsistency. Second, we put forward the Large margin Distance-weighted kNN LDL (LDkNN-LDL) that learns distance-dependent weight vectors to consider the difference in the neighborhoods of different instances.


Use our code and cite
>@article{wang_large_2023, \
	title = {Large Margin Weighted k-Nearest Neighbors Label Distribution Learning for Classification}, \
	journal = {IEEE Transactions on Neural Networks and Learning Systems},\
	author = {Wang, Jing and Geng, Xin},\
	year = {2023},\
  	doi={10.1109/TNNLS.2023.3297261},
>}



# How to use

run LWkNN-LDL: python run_dweighted_knn.py
run LDkNN-LDL: python run_weighted_knn.py


