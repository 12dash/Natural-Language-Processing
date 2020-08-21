import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from models import *


def generate_text(model, start_string, char2idx, idx2char):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def get_model(vocab_size, embedding_dim, rnn_unit, c_dir):
  model = build_gru(vocab_size, embedding_dim, rnn_unit, batch_size=1)
  model.load_weights('saved_model/model.h5')
  model.build(tf.TensorShape([1, None]))
  return model

def get_prediction(vocab_size, embedding_dim, rnn_unit, char2idx, idx2char):
  model = get_model(vocab_size, embedding_dim, rnn_unit, 'Model Checkpoint')
  a = (generate_text(model, start_string=u"सुरीली सोच में",char2idx = char2idx, idx2char = idx2char))
  file2write=open("Output.txt",'w',encoding='utf8')
  file2write.write(a)
  file2write.close()

