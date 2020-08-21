from preprocess import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt
from models import *

def train_model(dataset,vocab,embedding_dim,rnn_unit,batch_size):
  model = build_gru(
      vocab_size = len(vocab),
      embedding_dim=embedding_dim,
      rnn_units=rnn_unit,
      batch_size=batch_size)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam', loss=loss)

  history = model.fit(dataset, epochs = 100)
  model.save('saved_model/model.h5') 

  return model
