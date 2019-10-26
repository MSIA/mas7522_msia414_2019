import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
import os
import re
from gensim.models import Word2Vec

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
    with open('output_3.txt', 'w') as f:
        for i in normalized_corpus:
            for j in i:
                f.write('%s ' % j)
            f.write('\n\n')
    return normalized_corpus

def test_word2vec(normalized_corpus):
    for i in [0,1]:
        for j in [10, 100, 200]:
            word2vec_model = Word2Vec(normalized_corpus, size=j, window=5, min_count=5, workers=5, sg=i)
            print('Model: ' + str(i) + 'Size: ' + str(j))
            print(word2vec_model.wv.most_similar('happy'))
            print(('\n\n'))
            print(word2vec_model.wv.most_similar('truth'))
            print(('\n\n'))
            print(word2vec_model.wv.most_similar('schedule'))
            print(('\n\n'))
            print(word2vec_model.wv.most_similar('time'))
            print(('\n\n'))
            print(word2vec_model.wv.most_similar('friend'))

if __name__ == "__main__":

    #get all subfolders with newsgroup data
    d='20_newsgroups'
    folders = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
    print(folders)

    #Creating corpus
    fulltext = []
    for i in range(len(folders)):
        folder = '20_newsgroups/' + folders[i]
        files = list(filter(lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)))
        for j in range(len(files)):
            file1 = open(folder + '/' + files[j], "r+", encoding="latin-1")
            text = file1.read()
            fulltext.append(text)
            file1.close()
    
    normalized_corpus = preprocess(fulltext)
    test_word2vec(normalized_corpus)
