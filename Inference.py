#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:55:02 2020

@author: leo
"""




import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold, GroupKFold


import numpy as np
from scipy import stats

VERSION = 2
SEED = 111

NO_CLASSES = 20
N_FOLDS = 5
TRAIN_LENGTH = 200000
MAX_LENGTH = 100
BATCH_SIZE = 256

train = pd.read_csv('data/Train.csv')

train = train[train['CREATURE'] != 'creature4']

train = train.iloc[:TRAIN_LENGTH, :].reset_index(drop=True)

print(f"Train: {train.shape}")
train['seq_char_count'] = train['SEQUENCE'].apply(lambda x: len(x))

codes = {code for seq in train['SEQUENCE'] for code in seq}

#codes = [i for i in codes if i not in ['X', 'B', 'Z', 'U']]

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)


print(char_dict)
print("Dict Length:", len(char_dict))


def integer_encoding(data):
  """
  - Encodes code sequence to integer values.
  - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
  """
  
  encode_list = []
  for row in data['SEQUENCE'].values:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list
  
train_encode = integer_encoding(train) 

train_pad = pad_sequences(train_encode, maxlen=1000, padding='post', truncating='post')


test = pd.read_csv('data/Test.csv')


test_encode = integer_encoding(test) 

test_pad = pad_sequences(test_encode, maxlen=MAX_LENGTH, padding='post', truncating='post')

print(test_pad.shape)

X_test = to_categorical(test_pad)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test)).batch(BATCH_SIZE).prefetch(1)


del X_test, train_pad, train_encode, train, test

preds_list = []
for i in range(1, 6):
      new_model = tf.keras.models.load_model(f'models/ProtCNN-v{VERSION}.{i}.h5')
      preds = new_model.predict(test_dataset, verbose=1)
      preds_list.append(preds)
      print(f"Model no. {i} complete")
      

res = np.array([np.argmax(i, axis=1, out=None) for i in preds_list])

res_mode = stats.mode(res)
      


submission = pd.read_csv('data/SampleSubmission.csv')

submission['LABEL'] = [f'class{i}' for i in res_mode[0][0]]

submission.to_csv(f'results/submission_{VERSION}.csv', index=False)



      

