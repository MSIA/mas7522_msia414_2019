import pickle
import json
import pandas as pd
import numpy as np

def prediction(data, inputtext):
    mod_svm = pickle.load(open('mod.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    
    pred = tfidf.transform(data['reviewText'])
    svm_pred = mod_svm.predict(pred)
    
    json_dict = {"Review": inputtext, "Quality": str(svm_pred[0])}
    output = json.dumps(json_dict)
    return output    
    
if __name__ == "__main__":
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')
    input_1 = 'What the heck I hated this product. I cannot believe it even exists. terrible'
    input_2 = 'This is a product hand crafted by Jesus, a beautiful thing that gave my life hope.'
    input_3 = 'Everything in the world is made worse by this thing. I review it poorly.'
    input_4 = 'A wonderful thing for every household. My daughter is obsessed. Happy times.'
    output_1 = prediction(data, input_1)
    output_2 = prediction(data, input_2)
    output_3 = prediction(data, input_3)
    output_4 = prediction(data, input_4)
    with open('predictoutput_svm.txt', 'a') as f:
        print('Input: %s' % input_1, file = f)
        print('Output %s' % output_1, file = f)
        print('Input: %s' % input_2, file = f)
        print('Output %s' % output_2, file = f)
        print('Input: %s' % input_3, file = f)
        print('Output %s' % output_3, file = f)
        print('Input: %s' % input_4, file = f)
        print('Output %s' % output_4, file = f)
        