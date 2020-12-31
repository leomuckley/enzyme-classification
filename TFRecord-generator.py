#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:46:28 2020

@author: leo
"""
import time, json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.python.lib.io.tf_record import TFRecordWriter

train = pd.read_csv('Train.csv')

test = pd.read_csv('Test.csv')


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
  

def clean_train(csv):
    csv['LABEL'] = csv['LABEL'].str.replace('[A-Za-z]', '').astype(int)
    csv['SEQUENCE'] = csv['SEQUENCE'].apply(lambda x: str(x)[:400])
    train_encode = integer_encoding(csv) 
    train_pad = pad_sequences(train_encode, maxlen=400, padding='post', truncating='post')
    csv['SEQUENCE'] = [list(i) for i in train_pad]
    return csv


def clean_test(csv):
    #csv['LABEL'] = csv['LABEL'].str.replace('[A-Za-z]', '').astype(int)
    csv['SEQUENCE'] = csv['SEQUENCE'].apply(lambda x: str(x)[:400])
    test_encode = integer_encoding(csv) 
    test_pad = pad_sequences(test_encode, maxlen=400, padding='post', truncating='post')
    test_pad = pad_sequences(test_encode, maxlen=400, padding='post', truncating='post')
    csv['SEQUENCE'] = [list(i) for i in test_pad]
    return csv


train = clean_train(train)
test = clean_test(test)


# @tf.function(jit_compile=True)
tf.compat.v1.enable_eager_execution()
def create_tf_example(idx, features, label):
    # Examples maps string keys to either ints, floats or bytes
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
        # Sequence is stored in UTF-8 bytes
        'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        # Group is the creature the sample came from and is stored in UTF-8 bytes
        'group': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[2].encode('utf-8')])),
        # Label is the class we are trying to predict
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))

    # Encoded example can be extracted strings, 
    return tf_example


def convert_train_to_tfrecord(csv, file_name):
    start_time = time.time()
    writer = TFRecordWriter(file_name)
    for index, row in csv.iterrows():
        try:
            if row is None:
                raise Exception('Row Missing')
            if row[0] is None or row[1] is None or row[2] is None:
                raise Exception('Value Missing')
            if row[1].strip() == "":
                raise Exception('Sequence is empty')
            features, label = row[:-1], row[-1]
            example = create_tf_example(index, features, label)
            writer.write(example.SerializeToString())
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
    writer.close()
    print(f"{file_name}: --- {(time.time() - start_time)} seconds ---")
    

def convert_test_to_tfrecord(csv, file_name):
    start_time = time.time()
    writer = TFRecordWriter(file_name)
    for index, row in csv.iterrows():
        try:
            if row is None:
                raise Exception('Row Missing')
            if row[0] is None or row[1] is None or row[2] is None:
                raise Exception('Value Missing')
            if row[1].strip() == "":
                raise Exception('Sequence is empty')
            features, label = row, ""
            example = create_tf_example(index, features, label)
            writer.write(example.SerializeToString())
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
    writer.close()
    print(f"{file_name}: --- {(time.time() - start_time)} seconds ---")
    
    
def generate_json_info(local_file_name):
    info = {"train_length": len(train),
    # , "validation_length": len(validate_df),
             "test_length": len(test)}

    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)
    
    
convert_train_to_tfrecord(train, "data/enzyme_train.tfrecord")

convert_test_to_tfrecord(test, "data/enzyme_test.tfrecord")

generate_json_info("data/enzyme.json")
    


    
   
# tr_ds = tf.data.TFRecordDataset("data/enzyme.tfrecord")        
# # The dataset for train information

# feature_spec = {
#     'idx': tf.io.FixedLenFeature([], tf.int64),
#     'sequence': tf.io.FixedLenFeature([], tf.string),
#     'group': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.int64)
# }

# def parse_example(example_proto):
#   # Parse the input tf.Example proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_spec)


# tr_parse_ds = tr_ds.map(parse_example)

# dataset_iterator = iter(tr_parse_ds)

# dataset_iterator.get_next()





seq = train['SEQUENCE'].head()


seq = np.array([np.array(i) for i in seq])

tf_seq = tf.convert_to_tensor(seq)


one_seq = tf.one_hot(tf_seq, 25)


























    
