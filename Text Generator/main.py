import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("Importing the Preprocess python script")
from preprocess import *
print("Importing the Training python script")
from train import *
print("Importing the Predict python script")
from predictions import *
import tensorflow as tf

path = "Data/"+input("Enter the file name in the data folder including the prefix (.txt) : ")

print("Fetching the data ...")
text,vocab = get_text(path)

batch_size,buffer_size,rnn_unit, embedding_dim, vocab_size = set_up_params(vocab)

char2idx,idx2char,text_as_int,dataset = convert_data(text)

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)

model = train_model(dataset,vocab,embedding_dim,rnn_unit,batch_size)
model.summary()

get_prediction(vocab_size, embedding_dim, rnn_unit,char2idx,idx2char)