from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import sklearn
from nltk.corpus import stopwords
from tools import print_confusion_matrix
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd  
import os
import re
import random

def review_to_wordlist( review, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #

    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #review_text = review
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

###**********combine datasets for word2vec training************###
def word2vecInput(train,test,gold,yelp, shuffle=True, remove_stopwords=True):    
    #initialized train and gold test sentences
    goldSet  = gold["review"].tolist()
    trainSet = train["review"].tolist()
    testSet  = test["review"].tolist()
    
    trainSentences = yelp+trainSet+testSet
    print "Total Train Sentences: ",len(trainSentences)
    print "Total Gold Test Sentences: ", len(goldSet)
    
    #cleaning sentences    
    trainingSentences = [review_to_wordlist(review,remove_stopwords) for review in trainSentences]
    goldSentences     = [review_to_wordlist(review,remove_stopwords) for review in goldSet]
    #vocablary = trainingSentences+goldSentences
    #print "Total Vocablary of Sentences: ", len(vocablary)
    
    #Shuffle training sentences Sentences             
    if (shuffle):
        random.shuffle(trainingSentences)
        
    return trainingSentences#, vocablary

def splitDataset(examples, labels, split=0.8):  
    data = zip(examples,labels)
    random.shuffle(data)
    data_length = int(round(split*len(data)))
    train = data[:data_length] #training set
    validation = data[data_length:]#validation set
    train = zip(*train)
    validation = zip(*validation)
    x_train = pd.Series(np.array(train[0]))
    y_train = pd.Series(np.array(train[1]))
    x_val = pd.Series(np.array(validation[0]))
    y_val = pd.Series(np.array(validation[1]))
    return x_train, y_train, x_val, y_val

def tokenizerCleaner(yelp, remove_stopwords=True):
    print "Total Train Sentences: ",len(yelp)    
    yelp = [review_to_wordlist(review,remove_stopwords) for review in yelp]  
    return yelp

def oneHotVectors(labels):
    labels = pd.Series(labels)
    labels = pd.get_dummies(labels)
    labels.to_csv("Labels.csv")
    labels = labels.values
    return labels

def categoriesToLabels(labels):
    labels = pd.Series(labels)
    labels = pd.get_dummies(labels)
    labels = labels.values #copying the vectors for each category [1...13] length
    item_index = np.where(labels[:]==1)
    labels = item_index[1]
    return labels

def nextBatch(X_train, y_train, batch_size):
    sample = random.sample(zip(X_train,y_train), batch_size)
    sample = zip(*sample)
    return sample[0], sample[1]

def weight_variable(fan_in, fan_out, filename, boolean=False):   
    initial=0
    if (boolean):
        stddev = np.sqrt(2.0/fan_in)
        initial  = tf.random_normal([fan_in,fan_out], stddev=stddev)
    else:
        initial  = np.loadtxt(filename).astype(np.float32)
        #print initial.shape
    return tf.Variable(initial)

def resetModel():
    files = glob.glob('params/*')
    for f in files:
        os.remove(f)
    
def bias_variable(shape, filename, boolean=False):
    initial=0
    if (boolean):
        initial = tf.constant(0.1, shape=shape)
    else:
        initial  = np.loadtxt(filename).astype(np.float32) 
        #print initial.shape
    return tf.Variable(initial)
    
def confusionMatrix(y_pred, y_actu):
    return print_confusion_matrix(y_pred,y_actu)

def normalize(probs):
    prob_factor = 1 / np.sum(probs)
    return [prob_factor * p for p in probs]
    
def clean(sentences, remove_stopwords=False):
    sentences = [review_to_wordlist(review,remove_stopwords) for review in sentences]
    return sentences