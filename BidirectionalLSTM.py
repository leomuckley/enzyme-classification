#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:00:37 2020

@author: leo
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import L2

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold


SEED = 123

NO_CLASSES = 20
N_FOLDS = 3
MAX_LENGTH = 100


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


train = train.iloc[:10240, :]

train['seq_char_count'] = train['SEQUENCE'].apply(lambda x: len(x))

codes = {code for seq in train['SEQUENCE'] for code in seq}


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

train_pad = pad_sequences(train_encode, maxlen=MAX_LENGTH, padding='post', truncating='post')

print(train_pad.shape)


# One hot encoding of sequences
X = to_categorical(train_pad)

print(X.shape) 


y = train['LABEL'].str.replace('[A-Za-z]', '').astype(int)



## Define model
x_input = tf.keras.Input(shape=(None,), name='Input')

emb = tf.keras.layers.Embedding(25, 128, input_length=MAX_LENGTH)(x_input)
bi_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_regularizer=L2(0.01), 
                                                            recurrent_regularizer=L2(0.01), 
                                                            bias_regularizer=L2(0.01)))(emb)

# bi_rnn = tf.keras.layers.LSTM(64, kernel_regularizer=L2(0.01), 
#                               recurrent_regularizer=L2(0.01), 
#                               bias_regularizer=L2(0.01))(emb)

x = tf.keras.layers.Dropout(0.3)(bi_rnn)

# softmax classifier
x_output = tf.keras.layers.Dense(NO_CLASSES, activation='softmax')(x)

#model = tf.keras.Model(inputs=x_input, outputs=x_output)

model = tf.keras.Model(inputs=x_input, outputs=x_output, name="Model_1")


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

skf = StratifiedKFold(n_splits=N_FOLDS)

score_list = []
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    history = model.fit(X_train, y_train, epochs=50, batch_size=256,
                        validation_data=(X_val, y_val), callbacks=[es])
    best_score = max(history.history['acc'])
    score_list.append(best_score)

    







