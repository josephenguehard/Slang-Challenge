# -*-coding:Latin-1 -*
import numpy as np
import pandas as pd
import re
from nltk import WordNetLemmatizer
import copy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import math

""" N-Gram for characters """

class N_gram:
    """ n-gram of chars """
    
    ##### TODO: stop words english
    
    def __init__(self, n_range = [4, 5], max_df = 0.5, analyzer = 'char'):
        self.n_range = n_range
        self.max_df = 0.5
        self.word2docs = {}
        
        self.vectorizer = TfidfVectorizer(ngram_range = n_range, sublinear_tf=True, analyzer = analyzer, max_df=0.5, stop_words='english')

    def tf(self, word, document):
        
        f = document.count(word)
        
        if f == 0:
            return 0
        else:
            return math.log(f) + 1
            
    def idf(self, word):
        
        n_documents_with_word = len(self.word2docs[word])
        
        if n_documents_with_word == 0:
            return 0

        idf = (float(self.n_documents) + 1) / (n_documents_with_word + 1)
        idf = 1 + math.log(idf)
        return idf
        
    def tf_idf(self, word, document):
        tf = self.tf(word, document)
        if word in self.idf_keys:
            idf = self.idf_keys[word]
        else:
            idf = 0

        return tf * idf # CHECK IF WE NEED LOG
        
    def get_n_grams(self, document):
        n_range = self.n_range
        string = ' '.join(document) # PERHAPS PUT NOTHING INSTEAD OF SPACES
        
        result = []
        
        for n in n_range:
            
            result += [string[i:i+n] for i in range(len(string)-n+1)]
        
        return result
    
    def fit(self, X):      
        """vectorizer = self.vectorizer
        
        X_string = [' '.join(X[i]) for i in range(len(X))]
        
        vectorizer.fit(X_string)"""
        
        self.n_documents = len(X)
        
        for i in range(len(X)):
            x = X[i]
            n_grams = self.get_n_grams(x)
            
            for n_gram in n_grams:
                if n_gram in self.word2docs:
                    if i not in self.word2docs[n_gram]:
                        self.word2docs[n_gram].append(i)
                else:
                    self.word2docs[n_gram] = [i]
                    
        keys = list(self.word2docs.keys())
        self.keys = []
        
        self.idf_keys = {}
        
        for key in keys:
            idf = self.idf(key)
                
                
            df = len(self.word2docs[key])
            if float(df) / self.n_documents > self.max_df:
               continue
            
            self.idf_keys[key] = idf
            self.keys.append(key)
            
            
        self.keys.sort()
        
        
    def transform(self, X):     
        """vectorizer = self.vectorizer
        
        X_string = [' '.join(X[i]) for i in range(len(X))]
        
        X_out = vectorizer.transform(X_string)"""
        
        keys = self.keys
        
        #print(keys)
        
        
        X_out = sp.sparse.lil_matrix((len(X), len(keys)))
        for i_doc in (range(len(X))):
            document = X[i_doc]
            
            n_grams = self.get_n_grams(document)

            i_word = 0

            for word in keys:
                X_out[i_doc, i_word] = self.tf_idf(word, n_grams)
                i_word += 1
        
        norm = sp.sparse.linalg.norm(X_out, ord = 2, axis = 1)

        for i in range(X_out.shape[0]):
            if norm[i] > 0:
                X_out[i] /= norm[i]
        
        return X_out.tocsr()
        
    def fit_transform(self, X):
        
        self.fit(X)
        return self.transform(X)