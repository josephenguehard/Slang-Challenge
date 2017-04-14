# -*-coding:Latin-1 -*
import numpy as np
from preprocessing import preprocessing
from classification import Classifier
import pandas as pd
import re
from nltk import WordNetLemmatizer
import copy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from scipy.optimize import minimize


""" Detection of insults in comments """

def load_data(train_fname, test_fname):
    """ Load the train data and test data """

    X = []
    y = []
    
    with open(train_fname) as f:
        for line in f:
            y.append(int(line[0]))
            X.append(line[5:-4])
            
    y = np.array(y)
    
    for i in range(0,len(y)):
        if(y[i]==0):
            y[i]=-1
    
    X_test = []
    with open(test_fname) as f:
        for line in f:
            X_test.append(line[3:-4])
    
    return X, y, X_test


if __name__ == "__main__":
    
    # Load the data
    print("Loading data")
    
    X, y, X_real_test = load_data("train.csv", "test.csv")
    
    
    # Preprocessing
    
    print("Beginning preprocessing")
    
    X_processed, X_test_processed = preprocessing(X, X_real_test)

    assert X_processed.shape[1] == X_test_processed.shape[1] # check that train and test have the same number of features
    
    # Classification
    
    print("Beginning prediction")
    
    clf = Classifier(C=50000)

    X_train = X_processed.tocsr()
    y_train = y
    clf.fit(X_train, y_train)
    
    print("Score: %f" % clf.score(X_test, y_test))
    
    
    print("Saving prediction to file")
    
    y_pred = clf.predict(X_test_processed)
    
    # Save predictions to file
    
    np.savetxt('y_pred.txt', y_pred, fmt='%s')
