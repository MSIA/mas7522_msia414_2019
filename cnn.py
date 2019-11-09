import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import re
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


def train(train_texts, train_labels, dictionary, batch_size, n_layers, dropout_rate, model_file=None, EMBEDDINGS_MODEL_FILE=None):
    """
    Train a word-level CNN text classifier.
    :param train_texts: tokenized and normalized texts, a list of token lists, [['sentence', 'blah', 'blah'], ['sentence', '2'], .....]
    :param train_labels: the label for each train text
    :param dictionary: A gensim dictionary object for the training text tokens
    :param model_file: An optional output location for the ML model file
    :param EMBEDDINGS_MODEL_FILE: An optinal location for pre-trained word embeddings file location
    :return: the produced keras model, the validation accuracy, and the size of the training examples
    """
    assert len(train_texts)==len(train_labels)
    # compute the max sequence length
    # why do we need to do that?
    lengths=list(map(lambda x: len(x), train_texts))
    a = np.array(lengths)
    MAX_SEQUENCE_LENGTH = int(np.percentile(a, SEQUENCE_LENGTH_PERCENTILE))
    # convert all texts to dictionary indices
    #train_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), train_texts))
    train_texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
    # pad or truncate the texts
    x_data = pad_sequences(train_texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))
    # convert the train labels to one-hot encoded vectors
    train_labels = keras.utils.to_categorical(train_labels)
    y_data = train_labels

    model = Sequential()

    # create embeddings matrix from word2vec pre-trained embeddings, if provided
    if pretrained_embedding:
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
        embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
        for word, i in dictionary.token2id.items():
            embedding_vector = embeddings_index[word] if word in embeddings_index else None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=TRAINABLE_EMBEDDINGS))
    else:
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
    # add drop out for the input layer, why do you think this might help?
    model.add(Dropout(dropout_rate))
    # add a 1 dimensional conv layer
    # a rectified linear activation unit, returns input if input > 0 else 0
    model.add(Conv1D(filters=n_filters,
                     kernel_size=window_size,
                     activation='relu'))
    # add a max pooling layer
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
    model.add(Flatten())

    # add 0 or more fully connected layers with drop out
    for _ in range(n_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units,
                        activation=dense_activation,
                        kernel_regularizer=l2(l2_penalty),
                        bias_regularizer=l2(l2_penalty),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))

    # add the last fully connected layer with softmax activation
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(train_labels[0]),
                    activation='softmax',
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

    # compile the model, provide an optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print a summary
    print(model.summary())


    # train the model with early stopping
    early_stopping = EarlyStopping(patience=patience)
    Y = np.array(y_data)

    fit = model.fit(x_data,
                    Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=[early_stopping])

    print(fit.history.keys())
    val_accuracy = fit.history['acc'][-1]
    print(val_accuracy)
    # save the model

    if model_file:
        model.save(model_file)
    return model, val_accuracy, len(train_labels)

if __name__ == "__main__":

    # model hyper parameters
    EMBEDDING_DIM = 100
    SEQUENCE_LENGTH_PERCENTILE = 90
    n_layers = [1,2]
    hidden_units = 500
    batch_size = [100, 200]
    pretrained_embedding = False
    # if we have pre-trained embeddings, specify if they are static or non-static embeddings
    TRAINABLE_EMBEDDINGS = True
    patience = 2
    dropout_rate = [0.3, 0.5]
    n_filters = 100
    window_size = 8
    dense_activation = "relu"
    l2_penalty = 0.0003
    epochs = 10
    VALIDATION_SPLIT = 0.1

    #create mydict off stemmed_tokens
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')     
    corpus = data['reviewText'].tolist()
    processed_corpus = preprocess(corpus)
    ratings = [1 if i=='good' else 0 for i in data['quality']]
    mydict = gensim.corpora.Dictionary(processed_corpus)

    #experiment for different parameters and save models, print model accuracy to window
    for i in dropout_rate:
        for j in batch_size:
            for k in n_layers:
                train(processed_corpus, ratings, mydict, n_layers = k, batch_size=j, dropout_rate=i)

    #found best combo to be 0.3 dropout rate, 200 batch size, 1 layer
    
    train(processed_corpus, ratings, mydict, n_layers = 1, batch_size = 200, dropout_rate = 0.3, model_file='bestcnn.model')