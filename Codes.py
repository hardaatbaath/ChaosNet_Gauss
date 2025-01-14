# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
ChaosNet decision function
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC

import ChaosFEX.feature_extractor as CFX


def chaosnet(traindata, trainlabel, testdata):
    '''
    

    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label

    '''
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
        
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis = 1)

    return mean_each_class, predicted_label




def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, D_alpha, D_beta, EPSILON, DATA_NAME):
    """

    Parameters
    ----------
    FOLD_NO : TYPE-Integer
        DESCRIPTION-K fold classification.
    traindata : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    trainlabel : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    testlabel : TYPE - numpy 2D array
        DESCRIPTION - Testlabel
    INITIAL_NEURAL_ACTIVITY : TYPE - numpy 1D array
        DESCRIPTION - initial value of the chaotic skew tent map.
    DISCRIMINATION_THRESHOLD : numpy 1D array
        DESCRIPTION - thresholds of the chaotic map
    EPSILON : TYPE numpy 1D array
        DESCRIPTION - noise intenity for NL to work (low value of epsilon implies low noise )
    DATA_NAME : TYPE - string
        DESCRIPTION.

    Returns
    -------
    FSCORE, Q, B, EPS, EPSILON

    """
    ACCURACY = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    FSCORE = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    Q = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    A = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    B = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    EPS = np.zeros((len(D_alpha), len(D_beta), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))


    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) # Define the split - into 2 folds 
    KF.get_n_splits(traindata) # returns the number of splitting iterations in the cross-validator
    print(KF) 
    
    for alpha_idx, alpha in enumerate(D_alpha):
        for beta_idx, beta in enumerate(D_beta):
            for ina_idx, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                for eps_idx, EPSILON_1 in enumerate(EPSILON):
                    
                    ACC_TEMP =[]
                    FSCORE_TEMP=[]
                
                    for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                        
                        X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                        Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
            

                        # Extract features
                        FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, alpha, beta)
                        FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, alpha, beta)            
                    
                    
                        mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN,Y_TRAIN, FEATURE_MATRIX_VAL)
                        
                        ACC = accuracy_score(Y_VAL, Y_PRED)*100
                        RECALL = recall_score(Y_VAL, Y_PRED , average="macro", zero_division=1)
                        PRECISION = precision_score(Y_VAL, Y_PRED , average="macro", zero_division=1)
                        F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                                    
                        
                        ACC_TEMP.append(ACC)
                        FSCORE_TEMP.append(F1SCORE)
                    Q[alpha_idx, beta_idx, ina_idx, eps_idx ] = INA # Initial Neural Activity
                    A[alpha_idx, beta_idx, ina_idx, eps_idx ] = alpha # Alpha
                    B[alpha_idx, beta_idx, ina_idx, eps_idx ] = beta # Discrimination Threshold
                    EPS[alpha_idx, beta_idx, ina_idx, eps_idx ] = EPSILON_1 
                    ACCURACY[alpha_idx, beta_idx, ina_idx, eps_idx ] = np.mean(ACC_TEMP)
                    FSCORE[alpha_idx, beta_idx, ina_idx, eps_idx ] = np.mean(FSCORE_TEMP)
                    print("Mean F1-Score for Q = ", Q[alpha_idx, beta_idx, ina_idx, eps_idx ],"A = ", A[alpha_idx, beta_idx, ina_idx, eps_idx ],"B = ", B[alpha_idx, beta_idx, ina_idx, eps_idx ],"EPSILON = ", EPS[alpha_idx, beta_idx, ina_idx, eps_idx ]," is  = ",  np.mean(FSCORE_TEMP)  )
        
    print("Saving Hyperparameter Tuning Results")
    
    
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/SR-PLOTS/'  + DATA_NAME + '/NEUROCHAOS-RESULTS/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    np.save(RESULT_PATH+"/h_fscore.npy", FSCORE )    
    np.save(RESULT_PATH+"/h_accuracy.npy", ACCURACY ) 
    np.save(RESULT_PATH+"/h_Q.npy", Q ) 
    np.save(RESULT_PATH+"/h_A.npy", A )
    np.save(RESULT_PATH+"/h_B.npy", B )
    np.save(RESULT_PATH+"/h_EPS.npy", EPS )               
    
    
    MAX_FSCORE = np.max(FSCORE)
    Q_MAX = []
    A_MAX = []
    B_MAX = []
    EPSILON_MAX = []
    
    for alpha_idx in range(0, len(D_alpha)):
        for beta_idx in range(0, len(D_beta)):
            for ina_idx in range(0, len(INITIAL_NEURAL_ACTIVITY)):
                for WID in range(0, len(EPSILON)):
                    if FSCORE[alpha_idx, beta_idx, ina_idx, WID] == MAX_FSCORE:
                        Q_MAX.append(Q[alpha_idx, beta_idx, ina_idx, WID])
                        A_MAX.append(A[alpha_idx, beta_idx, ina_idx, WID])
                        B_MAX.append(B[alpha_idx, beta_idx, ina_idx, WID])
                        EPSILON_MAX.append(EPS[alpha_idx, beta_idx, ina_idx, WID])
    
    print("F1SCORE", FSCORE)
    print("BEST F1SCORE", MAX_FSCORE)
    print("BEST INITIAL NEURAL ACTIVITY = ", Q_MAX)
    print(f"BEST DISCRIMINATION THRESHOLD Alpha, Beta= {A_MAX}, {B_MAX} ")
    print("BEST EPSILON = ", EPSILON_MAX)
    return FSCORE, Q, A, B, EPS, EPSILON
    

