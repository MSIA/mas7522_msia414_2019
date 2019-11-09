import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import LinearSVC

def svm_model(data, outputtxt):
    loss = ['hinge', 'squared_hinge']
    C = [0.05, 0.5, 1]
    ngram_range = [(1, 1), (1, 2)]
    f1_list = []
    for i in ngram_range:
        tfidf = TfidfVectorizer(min_df=10, norm="l2", ngram_range=i, smooth_idf=True)
        tfidf.fit(data['reviewText'])
        preds = tfidf.transform(data['reviewText'])
        X_train, X_test, y_train, y_test = train_test_split(preds, data['quality'], train_size=0.80, test_size=0.20)
        for j in C:
            for k in loss:
                mod_svm = LinearSVC(C=j, random_state=1, loss = k)
                mod_svm.fit(X_train, y_train)
                preds = mod_svm.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, pos_label = 'good')
                rec = recall_score(y_test, preds, pos_label = 'good')
                f1 = (2*prec*rec)/(prec + rec)
                f1_list.append(f1)
                if outputtxt is not None:
                    with open(outputtxt, "a") as f:
                        print("Accuracy for C=%s, loss=%s and grams=%s: %s"
                            % (j, k, i, acc), file=f)
                        print("Precision: ", prec, file=f)
                        print("Recall: ", rec, file=f)
                        print("F1 score: ", f1, file=f)
    return np.argmax(f1_list)

def pickle_svm(data, argmax, modelpickl, tfidfpickl):
    if argmax in [0,1,2,3,4,5]:
        ngram_range = (1,1)
    else:
        ngram_range = (1,2)
    if argmax in [0,1,6,7]:
        C = 0.05
    elif argmax in [2,3,8,9]:
        C = 0.5
    else:
        C = 1
    if argmax in [0,2,4,6,8,10]:
        loss = 'hinge'
    else:
        loss = 'squared_hinge'
    tfidf = TfidfVectorizer(min_df=10, norm="l2", ngram_range=ngram_range, smooth_idf=True)
    tfidf.fit(data['reviewText'])
    preds = tfidf.transform(data['reviewText'])
    X_train, X_test, y_train, y_test = train_test_split(preds, data['quality'], train_size=0.80, test_size=0.20)
    mod_svm = LinearSVC(C=C, random_state=1, loss = loss)
    mod_svm.fit(X_train, y_train)
    with open(modelpickl, 'wb') as f:
        pickle.dump(mod_svm, f)
    with open(tfidfpickl, 'wb') as f:
        pickle.dump(tfidf, f)

if __name__ == "__main__":
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')
    argmax = svm_model(data, 'output2.txt')
    pickle_svm(data, argmax, 'mod.pkl', 'tfidf.pkl')