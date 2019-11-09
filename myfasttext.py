import pandas as pd
import numpy as np 
import fasttext
from sklearn.model_selection import train_test_split

def fast_text_prep(data, labelcol, textcol):
    fastdata = pd.DataFrame()
    fastdata['label'] = ['__label__' + str(i) for i in data[labelcol]]
    fastdata['text'] = data[textcol].str.replace('\n', ' ', regex=True)
    list_fastdata = [i + ' ' + j for i,j in zip(fastdata['label'].tolist(), fastdata['text'].tolist())]
    return fastdata, list_fastdata

def fasttext_model(data, outputtxt):
    loss = ['ns', 'hs', 'softmax']
    ngram_range = [1, 2]
    L = [0.01, 0.1, 0.5]
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], train_size=0.80, test_size=0.20)
    full_train = [i + ' ' + j for i,j in zip(y_train.tolist(), X_train.tolist())]
    full_test = [i + ' ' + j for i,j in zip(y_test.tolist(), X_test.tolist())]
    with open('train.txt', 'w') as f:
        for i in full_train:
            print(i, file = f)
    with open('test.txt', 'w') as f:
        for i in full_test:
            print(i, file = f)
    for i in ngram_range:
        for j in L:
            for k in loss:
                mod_ft = fasttext.train_supervised(input='train.txt', lr=j, wordNgrams=i, loss=k)
                metrics = mod_ft.test_label('test.txt')
                if outputtxt is not None:
                    with open(outputtxt, "a") as f:
                          print("Metrics for L=%s, loss=%s and grams=%s: %s"
                              % (j, k, i, metrics), file=f)

if __name__ == "__main__": 
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')                     
    fastdata, list_fastdata = fast_text_prep(data, 'quality', 'reviewText')
    fasttext_model(fastdata, 'output_fasttxt.txt')