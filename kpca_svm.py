#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

# Kernel PCA

class KPCA:
    ''' Kernel Principal Component Analysis
    
    Parameters
    ----------------------------------------------------------------------------------
    n_comps: init
     No of prinicpal components
    kernel: string, default = poly
     Type of kernel: rbf, poly
    degree: int, default = 2
     Degree of polynomial kernel
    gamma: int, default = 1
     Kernel coefficient for rbf kernel

    Attributes
    ------------------------------------------------------------------------------------
    K : Kernel matrix of input images (either rbf or poly kernels)
    v : Principal Components
    '''
    
    def __init__(self,n_comps,kernel='poly',degree=2,gamma=1):
        self.n_comps = n_comps
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
    def _rbfKernel(self,in_mtx):
        ''' Radial Basis Function
        Kernel is caluculated using the formula exp(-gamma*||x1-x2||^2)
        
        Parameters
        ---------------------------------------------------------------------------------------
        in_mtx: Input image matrix where each row represents flattened pixel values of an image
        
        Returns
        ----------------------------------------------------------------------------------------
        K: Kernel matrix (not centerd) for the input
        '''
        sqdist = cdist(in_mtx,self.X,'sqeuclidean') #return pairwise squared euclidean distances
        K = np.exp(-self.gamma * sqdist)
        return K
    def _polyKernel(self,in_mtx):
        ''' Polynomial Kernel
        Kernel is caluculated using the formula ((xi.x1+xi.x2+...+xi.xn)/n+1)^degree 
        i = 1 to #images of in_mtx, n = #images of train set
        
        Parameters
        ---------------------------------------------------------------------------------------
        in_mtx: Input image matrix where each row represents flattened pixel values of an image
        
        Returns
        ----------------------------------------------------------------------------------------
        K: Kernel matrix (not centerd) for the input
        '''
        K = (np.dot(in_mtx,self.X.T)/self.Nx+1)**self.degree
        return K
    def fit(self,X):
        '''Fit function
        Finds the specified #of principal components from the training set
        
        Parameters
        -----------------------------------------------------------------------------------------
        X: The input image matrix used to find the principal components
        '''
        self.X = X
        self.Nx = X.shape[0]  #no of samples
        if self.kernel=='poly':
            self.K =self. _polyKernel(X)
        if self.kernel=='rbf':
            self.K = self._rbfKernel(X)
        self.ones_mtx = np.ones((self.Nx,self.Nx))/self.Nx  #compute square matrix with all ones
        self.K = self.K - self.ones_mtx.dot(self.K) - self.K.dot(self.ones_mtx) + self.ones_mtx.dot(self.K).dot(self.ones_mtx) #centering the K matrix
        w,v = np.linalg.eig(self.K)
        v = np.real_if_close(v, tol=1)
        self.v = v[:,0:self.n_comps]  #select no. of principal components 
    def transform(self,X):
        '''Transform function
        Transforms the input matrix to feature space
        
        Parameters
        -----------------------------------------------------------------------------------------
        X: The input image matrix to be transformed
        
        Returns
        -----------------------------------------------------------------------------------------
        X_trans: The transformed input matrix to the feature space
        '''
        if self.kernel == 'poly':
            K_trans = self._polyKernel(X)
            print("K_trans:",self.K.shape)
        if self.kernel == 'rbf':
            K_trans = self._rbfKernel(X)
        N_trans = X.shape[0]
        ones_mtx_trans = np.ones((N_trans,self.Nx))/self.Nx
        K_trans = K_trans-np.dot(ones_mtx_trans,self.K)-np.dot(K_trans,self.ones_mtx)+ones_mtx_trans.dot(self.K).dot(self.ones_mtx)
        print("K_trans2:",K_trans.shape)
        X_trans = np.dot(K_trans,self.v)
        print("v:",self.v.shape)
        print("X_trans:",X_trans.shape)
        return X_trans  


def train_model(train_df):
    label = np.array(train_df["label"])
    image_arr = np.array(train_df.drop(["label"],axis=1))
    # Mean centering the image data
    image_arr_centrd = image_arr - np.mean(image_arr)

    # ## Dimensionality Reduction and Classification

    kpca = KPCA(n_comps = 10, kernel='poly', degree=2) #input the dimension in the reduced space, type of kernel and its parameters
    kpca.fit(image_arr_centrd)
    trainProjFace = kpca.transform(image_arr_centrd) # Transform the train and test image arrays to the dimensionality reduced feature space
    print(trainProjFace.shape)
    #testProjFace = kpca.transform(test_arr)

    # Create the parameter grid  
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # Performing CV to tune parameters for best SVM fit 
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(trainProjFace, label)
    best_C = svm_model.best_estimator_.C
    best_kernel = svm_model.best_estimator_.kernel
    best_gamma = svm_model.best_estimator_.gamma

    #Model generation using train set
    classifier = SVC(kernel=best_kernel, C=best_C , gamma=best_gamma)
    model = classifier.fit(trainProjFace, label)
    return kpca,model

def faceid(face_arr,kpca,model):
    projFace = kpca.transform(face_arr.reshape(1,11500))
    print("projFace:",projFace.shape)
    out_label = model.predict(projFace)
    return out_label


