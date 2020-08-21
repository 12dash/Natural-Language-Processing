import numpy as np
import time  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def get_text(path):
    text = open(path, 'rb').read().decode(encoding = 'utf-8')
    print("The number of characters in the file including the spaces and all the literals (such as ',\" \\n)")
    print(f"Number : {len(text)} characters")
    vocab = sorted(set(text))
    print("The number of unique character in the dataset : ",len(vocab))
    return text,vocab

def split_input_target(chunk):
    input_text = chunk[:-1]
    output_text = chunk[1:]
    return input_text, output_text

def convert_data(text):
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int  = np.array([char2idx[c] for c in text])
    max_length = 100
    seq_length = max_length
    examples_per_epoch = len(text)//(seq_length+1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)
    dataset = sequences.map(split_input_target)
    
    return char2idx,idx2char,text_as_int,dataset
