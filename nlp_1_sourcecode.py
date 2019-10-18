import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
import os
import spacy
from spacy.lang.en import English
import glob
import re
import datetime

#get all subfolders with newsgroup data
d='NLP_1/20_newsgroups'
folders = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
print(folders)

#Creating corpus
fulltext = []
for i in range(len(folders)):
    folder = 'NLP_1/20_newsgroups/' + folders[i]
    files = list(filter(lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)))
    for j in range(len(files)):
        file1 = open(folder + '/' + files[j], "r+", encoding="latin-1")
        text = file1.read()
        fulltext.append(text)
        file1.close()

time=datetime.datetime.now()
sentence_tokens = []
word_tokens = []
for document in fulltext:
    sentences = sent_tokenize(document)
    words = word_tokenize(document)
    for i in sentences:
        sentence_tokens.append(i)
    for j in words:
        word_tokens.append(j)
print("Time: %s"%(datetime.datetime.now()-time))

time=datetime.datetime.now()
stem = PorterStemmer()
corpus_stems = []
for i in word_tokens:
    corpus_stems.append(stem.stem(i))
print("Time: %s"%(datetime.datetime.now()-time))

time=datetime.datetime.now()
POS_tagged_corpus = nltk.pos_tag(word_tokens)
print("Time: %s"%(datetime.datetime.now()-time))

time=datetime.datetime.now()
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
spacy_sentence_tokens = []
spacy_word_tokens = []
for i in fulltext:
    file = nlp(i)
    s = [j.string.strip() for j in file.sents]
    w = [k.text for k in file]
    for j in s:
        spacy_sentence_tokens.append(j)
    for k in w:
        spacy_word_tokens.append(k)
print("Time: %s"%(datetime.datetime.now()-time))

time=datetime.datetime.now()
spacy_corpus_POS = []
for i in fulltext:
    file = nlp(i)
    for j in file:
        spacy_corpus_POS.append([j.text,j.pos_])
print("Time: %s"%(datetime.datetime.now()-time))

emails = []

for i in spacy_sentence_tokens:
        one = re.findall(r'[\w\.-]+@[\w-]+\.[\w\.-]+', i)
        for i in one:
            emails.append(i)

len(emails)
emails[0:10]

d1= '(Jan|Feb|Mar|Apr|May|Jun|July|Aug|Sept|Oct|Nov|Dec)\s(\d\d?).+?(\d\d\d\d)'
d2= '(January|February|March|April|May|June|July|August|September|October|November|December)\s(\d\d?).+?(\d\d\d\d)'
d3= '(\d\d?)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec),?\s+?(\d\d\d\d)'
d4= '(\d\d?)\s(January|February|March|April|May|June|July|August|September|October|November|December),?\s+?(\d\d\d\d)'
dates = []
for i in spacy_sentence_tokens:
    one = re.findall(d1, i) + re.findall(d2, i) + re.findall(d3, i) + re.findall(d4, i)
    for i in one:
        dates.append(i)

len(dates)
dates[0:10]