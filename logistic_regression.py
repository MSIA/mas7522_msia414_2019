import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def logistic_model(data, outputtxt):
    penalty = ['l1', 'l2'] 
    C = [0.05, 0.5, 1]
    ngram_range = [(1, 1), (1, 2)]
    for i in ngram_range:
        tfidf = TfidfVectorizer(min_df=10, norm="l2", ngram_range=i, smooth_idf=True)
        tfidf.fit(data['reviewText'])
        preds = tfidf.transform(data['reviewText'])
        X_train, X_test, y_train, y_test = train_test_split(preds, data['quality'], train_size=0.80, test_size=0.20)
        for j in C:
            for k in penalty:
                mod_log = LogisticRegression(C=j, solver="saga", random_state=1, penalty = k)
                mod_log.fit(X_train, y_train)
                preds = mod_log.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, pos_label = 'good')
                rec = recall_score(y_test, preds, pos_label = 'good')
                f1 = (2*prec*rec)/(prec + rec)
                if outputtxt is not None:
                    with open(outputtxt, "a") as f:
                        print("Accuracy for C=%s, penalty=%s and grams=%s: %s"
                            % (j, k, i, acc), file=f)
                        print("Precision: ", prec, file=f)
                        print("Recall: ", rec, file=f)
                        print("F1 score: ", f1, file=f)

if __name__ == "__main__":
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')
    logistic_model(data, 'output.txt')