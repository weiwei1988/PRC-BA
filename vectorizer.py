
# coding: utf-8

import re
import os
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

cur_dir = os.getcwd()
stop = pickle.load(open(os.path.join(cur_dir, 'movieclassifer', 'pkl_objects', 'stopwords.pkl'), 'rb'))
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)

