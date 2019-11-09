import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_json('reviews_Home_and_Kitchen_5.json', lines = True)
    data['quality'] = np.where(data['overall'] >=4, 'good', 'bad')
    reviewList = data.reviewText.tolist()
    reviewLen = [len(i) for i in reviewList]     
    with open('stats_output.txt', 'a') as f:
        print('Number of Documents: %s' % len(data.reviewText), file = f)
        print( 'Label Distribution: %s' % data.groupby('quality').count()['reviewText'], file = f)
        print('Number of Labels: %s' % len(data.quality.unique()), file = f)
        print('Mean Length of Review: %s' % np.mean(reviewLen), file = f)
