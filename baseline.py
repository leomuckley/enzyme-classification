#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:22:41 2020

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


SEED=123

TRAIN_LENGTH = 100000

train = pd.read_csv('data/Train.csv')

train = train.iloc[:TRAIN_LENGTH, :]

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

print(train_pad.shape)

del train_encode
y = train['LABEL'].str.replace('[A-Za-z]', '').astype(int)

y = to_categorical(y)
print(y.shape)

del train

# One hot encoding of sequences
X = to_categorical(train_pad)
print(X.shape) 



# One hot encoding of sequences
y = to_categorical(y)
print(X.shape) 

del train_pad


from sklearn.decomposition import PCA


pca = PCA(n_components=100, random_state=SEED)

X_pca = pca.fit_transform(X)



for i in X:
    print(np.argmax(X, axis=None, out=None))



t = np.argmax(y, axis=1, out=None)



t1



















