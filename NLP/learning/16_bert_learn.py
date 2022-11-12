import os
import re
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string],'')
    plt.xlabel('Epochs')
    plt.ylabel('string')
    plt.legend([string, 'va;_'+string])
    plt.show

tf.random.set_seed(1111)
np.random.seed(1111)

BLASS_NUMBER = 2
BATCH_SIZE = 32
NUM_EPOCHS = 2
VALID_SPLIT = 0.2
MAX_LEN = 40
BERT_CKPT = 'C:/pytest/data/KOR/BERT/bert_ckpt/'
DATA_IN_PATH = 'C:/pytest/data/KOR/naver_movie/data_in/'
DATA_OUT_PATH = 'C:/pytest/data/KOR/BERT/data_out'

def listToString(listdata):
    result = 'id\tdocument\tlabel\n'
    for data_each in listdata:
        if data_each:
            result += data_each[0]+'\t'+data_each[1]+'\t'+data_each[2]+'\n'
        return result
    
def read_data(filename, encoding='cp949', start=0):
    with open(filename, 'r', encoding=encoding) as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[start:]
    return data


def write_data(data, filename, encoding = 'cp949'):
    with open(filename, 'w', encoding=encoding)as f:
        f.write(data)
        
data_ratings = read_data(os.path.join(DATA_IN_PATH, 'ratings_utf8_small.txt'), encoding='utf-8',start=1)
data_ratings

from sklearn.model_selection import train_test_split
ratings_train, ratings_test = train_test_split(data_ratings)

ratings_train = listToString(ratings_train)
ratings_test = listToString(ratings_test)

write_data(ratings_train, os.path.join(DATA_IN_PATH, 'ratings_train.txt'), encoding = 'utf-8')
write_data(ratings_test, os.path.join(DATA_IN_PATH, 'ratings_test.txt'), encoding= 'utf-8')

tokenizer = BertTokenizer.from_pretrained('bert-basa-multilingual-cased', 
                                          cache_dir = os.path.join(BERT_CKPT, 'tokenizer'), 
                                          do_lower_case = False)

import pickle

if os.path.exists(DATA_OUT_PATH):
    print(f'{DATA_OUT_PATH} -- Folder already exists \n')
else:
    os.makedirs(DATA_OUT_PATH, exist_ok=True)
    print(f'{DATA_OUT_PATH} -- Folder create complete \n')
    
with open(DATA_OUT_PATH+'bert_tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)