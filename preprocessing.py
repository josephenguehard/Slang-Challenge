# -*-coding:Latin-1 -*
import numpy as np
import pandas as pd
import re
from nltk import WordNetLemmatizer
import copy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from n_gram import N_gram

""" Data preprocessing """

    
def bad_words(documents, wordlist_fname):
    """ Creates 2 features: count and ratio of bad words """
    
    wordnet_lemmatizer = WordNetLemmatizer()
    v_lemmatize = np.vectorize(wordnet_lemmatizer.lemmatize)
    
    def bad_words_text(document, wordlist_fname):
        """ Returns the number of bad_words and ratio of bad_words in a single example """
        
        document_length = len(document)
        
        if document_length == 0:
            return 0, 0
        
        badwords = pd.read_csv("badwords.txt", sep = '\n').as_matrix()
        badwords = np.ndarray.flatten(badwords)
        badwords = v_lemmatize(badwords) # lemmatize
        
        
        mask = np.in1d(document, badwords)
        
        count = len(mask[mask == True])
        ratio = float(count) / document_length
        
        return count, ratio
    
    v_bad_words = np.vectorize(bad_words_text)
    
    count, ratio = v_bad_words(documents, wordlist_fname)
    
    count = count.reshape((-1, 1))
    ratio = ratio.reshape((-1, 1))
    
    X = np.hstack([count, ratio])
    
    return X

def you(documents, wordlist_fname):
    """ Creates 2 features: count and ratio of bad words """
    
    wordnet_lemmatizer = WordNetLemmatizer()
    v_lemmatize = np.vectorize(wordnet_lemmatizer.lemmatize)
    
    def you_text(document, wordlist_fname):
        """ Returns the number of bad_words and ratio of bad_words in a single example """
        
        document_length = len(document)
        
        if document_length == 0:
            return 0, 0
        
        you = pd.read_csv("you.txt", sep = '\n').as_matrix()
        you = np.ndarray.flatten(you)
        you = v_lemmatize(you) # lemmatize
        
        
        mask = np.in1d(document, you)
        
        count = len(mask[mask == True])
        ratio = float(count) / document_length
        
        return count, ratio
    
    v_you = np.vectorize(you_text)
    
    count, ratio = v_you(documents, wordlist_fname)
    
    count = count.reshape((-1, 1))
    ratio = ratio.reshape((-1, 1))
    
    X = np.hstack([count, ratio])
    
    return X

def search_you(documents):
    """Create feature: distance of a you from a badword"""
    bad_words = 'badwords.txt'
    you = "you.txt"
    
    wordnet_lemmatizer = WordNetLemmatizer()

    BadWords = []
    with open(bad_words) as f:
        for line in f:
            BadWords.append(wordnet_lemmatizer.lemmatize(line[0:len(line)-1]))

    You = []
    with open(you) as f:
        for line in f:
            You.append(wordnet_lemmatizer.lemmatize(line[0:len(line)-1]))

    distance = np.zeros(len(documents))

    for i in range(0, len(documents)):
        is_badwords = []
        is_you = []
        for j in range(0, len(documents[i])):
            if(documents[i][j] in BadWords):
                is_badwords.append(j+1)
            if(documents[i][j] in You):
                is_you.append(j+1)

        score = 0
        if(len(is_badwords) and len(is_you)):
            pairs = list(itertools.product(is_badwords,is_you))
            for k in range(0, len(pairs)):
                score = score + 1./(abs(pairs[k][0] - pairs[k][1]))
        distance[i] = score         
        
    X = distance.reshape((-1, 1))
    
    return X
    
def uppercase_words(documents):
    """ Creates feature: ratio of uppercase words """
    
    def uppercase_words_text(document):
        """ Returns ratio of uppercase words in a single document """
        
        if len(document) == 0:
            return 0
        
        v_is_upper = np.vectorize(str.isupper)
        
        mask = v_is_upper(document)
        
        ratio = np.mean(mask == True)
        
        return ratio
            
    v_uppercase_words_text = np.vectorize(uppercase_words_text)
    
    X = v_uppercase_words_text(documents).reshape((-1, 1))
    
    return X

def exclamation_marks(documents):
    """ Creates one feature: ratio of exclamation marks """
    
    def exclamation_marks_text(document):
        """ Returns ratio of exclamation marks in a single document (compared to the number of words) """
        
        document_length = len(document)
        
        if document_length == 0:
            return 0
            
        count = 0
        
        for word in document:
            count += word.count('!')
        
        ratio = float(count) / document_length
        
        return ratio
    
    v_exclamation_marks_text = np.vectorize(exclamation_marks_text)
    
    X = v_exclamation_marks_text(documents).reshape((-1, 1))
    
    return X
    
def smileys(documents):
    """ Creates one feature: ratio of nice smileys """
    
    list_smileys = [':)', ':-)', ';)', ';-)', '=)', '=D', ':p', ':P', '<3']
    
    def smileys_text(document):
        """ Returns number of smileys in a single document """
        
        count = 0
        
        for word in document:
            for smiley in list_smileys:
                count += word.count(smiley)
        
        return count
        
    v_smileys_text = np.vectorize(smileys_text)
    
    X = v_smileys_text(documents).reshape((-1, 1))
    
    return X
    

def clean(f):
    """ Deletes noise such as \\n and replace some words like u --> you """
    
    f = [x.replace("\\n"," ") for x in f]        
    f = [x.replace("\\t"," ") for x in f]        
    f = [x.replace("\\xa0"," ") for x in f]
    f = [x.replace("\\xc2"," ") for x in f]
    f = [x.replace("\\xbf"," ") for x in f]
    f = [x.replace("\\r"," ") for x in f]
    f = [x.replace("\\"," ") for x in f]

    f = [x.replace(" u "," you ") for x in f]
    f = [x.replace(" em "," them ") for x in f]
    f = [x.replace(" da "," the ") for x in f]
    f = [x.replace(" yo "," you ") for x in f]
    f = [x.replace(" ur "," you ") for x in f]
    
    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]
    
    f = [x.replace("mudafucka", "motherfucker") for x in f]
    f = [x.replace("bytch", "bitch") for x in f]
    
    def remove_urls(document):
        """ Removes all urls in document. """
        
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', document)
        
        for url in urls:
            document = document.replace(url, " ")
            
        return document
    
    f = [remove_urls(x) for x in f]
    
    f = [x.replace(".", " ") for x in f]
    f = [x.replace("?"," ") for x in f]
    f = [x.replace("-"," ") for x in f]
    f = [x.replace("^"," ") for x in f]
    f = [x.replace("[", " ") for x in f]
    f = [x.replace("]", " ") for x in f]
    f = [x.replace(",", " ") for x in f]
    
    return f

def clean_twice(documents):
    """
    Deletes remaining noise such as '!' and ')'
    This function should be called when we are sure these symbols won't be usefull anymore
    """
    
    clean_documents = []
    
    for doc in documents:
        doc = [re.sub(r'[!\(\):;]',"", x) for x in doc] # ! ( ) : ;
        doc = [x for x in doc if x]
        
        if len(doc) == 0: # a document can't be empty, at least an empty string
            doc = ['']
        
        clean_documents.append(doc)
    
    return clean_documents

def separate(X):
    """ Create a list of each words of the sentence"""
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    X_separate = []
    for i in range(0,len(X)):
        if X[i] != []:
            lemmatized_vector = list(map(wordnet_lemmatizer.lemmatize, X[i].split()))# lemmatization

            X_separate.append(lemmatized_vector)
        
    return X_separate
    
def all_lowercase(documents):
    """ Make sure all letters are lowercase """
    
    lower_documents = []
    
    for doc in documents:

        lower_documents.append([x.lower() for x in doc])
        
    return lower_documents
    
class Tf_idf:
    """ TF IDF for vector of words X """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(sublinear_tf=False, analyzer = 'word', max_df=0.5, binary=False, stop_words='english')
    
    def fit(self, X):
        # TOUT RECODER
        
        # try binary = True, try sublinear_tf = True, max_df value
        vectorizer = self.vectorizer
        
        X_string = [' '.join(X[i]) for i in range(len(X))]
    
        vectorizer.fit(X_string)
        
    def transform(self, X):
        # TOUT RECODER
        
        # try binary = True, try sublinear_tf = True, max_df value
        vectorizer = self.vectorizer
        
        X_string = [' '.join(X[i]) for i in range(len(X))]
    
        X_tf_idf = vectorizer.transform(X_string)
        
        return X_tf_idf
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stop_words(X, filename = 'stopwords.txt'):
    """
        Remove stop words
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    v_lemmatize = np.vectorize(wordnet_lemmatizer.lemmatize)
    stopwords = pd.read_csv(filename, sep = '\n').as_matrix()
    stopwords = np.ndarray.flatten(stopwords)
    stopwords = v_lemmatize(stopwords) # lemmatize
    
    def stop_words_text(document):
        document = np.array(document)
        mask = np.in1d(document, stopwords)
        document = document[np.where(mask == False)]
        return document.tolist()
        
    X_stopped = [stop_words_text(doc) for doc in X]
    
    return X_stopped
        
def preprocessing(X, X_test):
    """ Create features """
    
    tf_idf = Tf_idf()
    n_gram = N_gram(n_range = (4, 5))

    def prepare(X, train = True):
        X = clean(X)
        X = separate(X)
        
        X_bad_words = bad_words(X, "badwords.txt")
        X_uppercase = uppercase_words(X)
        X_exclamation_marks = exclamation_marks(X)
        X_smileys = smileys(X)
        X_you = search_you(X)
        X_you2 = you(X, "you.txt")
        
        X = clean_twice(X)
        #X = stop_words(X) # does not improve the model
        X = all_lowercase(X)
        
        if train:
            X_tf_idf = tf_idf.fit_transform(X)
            X_n_gram = n_gram.fit_transform(X)
        else:
            X_tf_idf = tf_idf.transform(X)
            X_n_gram = n_gram.transform(X)
    
        X_processed = sp.sparse.hstack([X_bad_words, X_uppercase, X_exclamation_marks, X_smileys, X_you, X_you2, X_tf_idf, X_n_gram])
        #X_processed = sp.sparse.hstack([X_tf_idf, X_n_gram])
        
        return X_processed
        
    X_processed = prepare(X)
    X_test_processed = prepare(X_test, train = False)
    
    return X_processed, X_test_processed