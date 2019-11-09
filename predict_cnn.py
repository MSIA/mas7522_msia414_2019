import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import re
import json
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import gensim
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.models import load_model

def preprocess(corpus):
    tokenized_corpus = []
    for i in corpus:
        tokenized_document = word_tokenize(i)
        tokenized_corpus.append(tokenized_document)
    normalized_corpus = []
    for i in tokenized_corpus:
        normalized_document = []
        for j in i:
            j = re.sub(r'\d+', '', j) # remove numbers from corpus
            j = re.sub(r'[^\w\s]','', j.lower().strip()) #remove everything except words and space
            j = re.sub(r'\_','', j)
            normalized_document.append(j)
        normalized_document = [i for i in normalized_document if i] # remove empty string tokens
        normalized_document = [i for i in normalized_document if len(i) > 0]
        normalized_corpus.append(normalized_document)
    return normalized_corpus

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))

if __name__ == "__main__":
    theinput = ['What the heck I hated this product. I cannot believe it even exists. terrible', 'This is a product hand crafted by Jesus, a beautiful thing that gave my life hope.', 'Everything in the world is made worse by this thing. I review it poorly.', 'A wonderful thing for every household. My daughter is obsessed. Happy times.']
    tokens = preprocess(theinput)
    cnn = load_model('bestcnn.model')

    mydict = gensim.corpora.Dictionary.load('cnn.dict')
    train_texts_indices = list(map(lambda x: texts_to_indices(x, mydict), tokens))
    data = pad_sequences(train_texts_indices, maxlen=97)
    labels = cnn.predict_classes(data)
    labels = [int(i) for i in labels]
    probability = cnn.predict_proba(data)
    probability = [str(np.max(i)) for i in probability]

    with open('predictoutput_cnn.json', 'w') as f:
        json.dump({z[0]:list(z[1:]) for z in zip(theinput,labels,probability)}, f)