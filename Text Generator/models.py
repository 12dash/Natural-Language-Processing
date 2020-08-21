from preprocess import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt

def set_up_params(vocab):
  batch_size = 16 
  buffer_size = 10000
  rnn_unit = 1024
  embedding_dim = 256
  vocab_size = len(vocab)
  return batch_size,buffer_size,rnn_unit, embedding_dim, vocab_size

def build_gru(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
    return model

def build_lstm(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
     tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.LSTM(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer= 'glorot_uniform'),
     tf.keras.layers.Dropout(0.2), 
     tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
                          tf.keras.layers.Dropout(0.2),
                          tf.keras.layers.Dense(vocab_size)])
  return model
