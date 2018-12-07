# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:56:49 2017

@author: DH384961
"""
#import numpy as np
from sklearn import  datasets


# import some data to play with
iris = datasets.load_iris()
X = iris.data 
"""
from sklearn.decomposition import PCA
pca=PCA()
X_PCA= pca.fit_transform(X)

s=sorted(pca.explained_variance_, reverse=True)
sig=np.cumsum(s)/sum(s)
x=(np.where(sig>0.99))[0][1] # Upto x-th column of X_kpca for dimensionality reduction
print("x= ", x)
X_PCA=X_PCA[:,:x]
"""

# SImple k means
from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3, max_iter = 10,tol = 0.1,random_state = 1)

model.fit(X)

model.cluster_centers_
model.labels_

model.max_iter

"""
from sklearn.cluster import AgglomerativeClustering as hclust
model = hclust( linkage = 'ward',n_clusters=2)
model.fit(X)
model.labels_
"""

""" Evaluation of clusters and best K"""
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X = load_iris().data
#y = load_iris().target


inertia_kmeans = []
for k in range(2, 11):
    print ("K=", k )
    kmeans = KMeans(n_clusters=k,
                    max_iter = 10,
                    tol = 0.1,
                    random_state = 100)
    kmeans.fit(X)
    #hcluster= hclust(linkage="ward" ,  n_clusters=k).fit(X)
    label = kmeans.labels_; print ("Semicolon ") #label_h = hcluster.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    #sil_coeff_hclust = silhouette_score(X, label_h, metric='euclidean')
    inertia_kmeans.append(kmeans.inertia_)# Cost oR the Total Sum of Squares
    print ("costs: ", inertia_kmeans)
    print("For n_clusters={}, The H_clustering Silhouette Coefficient is {}".format(k, sil_coeff))
    #print("For n_clusters={}, The K_means Silhouette Coefficient is {}".format(k, sil_coeff))

# Elbow Plot
import matplotlib.pyplot as plot
plot.plot(range(2,11), inertia_kmeans)
plot.title("Elbow curve")
plot.ylabel("Inertia Or Sum of  all Within cluster Squares ")
plot.xlabel("No of Clusters")
  

    
    
    
    
    
    