import numpy as np
from parameter import *

import tensorflow as tf
import pickle

def avg(l):
    return sum(l)/len(l)


def load_pickle(name):
    path = os.path.join("pickle", name)
    return pickle.load(open(path,"rb"))


def save_pickle(name, obj):
    path = os.path.join("pickle", name)
    return pickle.dump(obj, open(path,"wb"))


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask


def glove_voca():
    path = path_dict["embedding_data_path"]
    voca = set()
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            voca.add(s[0])
    return voca


def load_embedding(word_indices, word_embedding_dimension, divident=1.0):
    print("loading embedding")
    path = path_dict["embedding_data_path"]
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    j = 0
    n = len(word_indices)
    m = word_embedding_dimension
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1, m), dtype="float32")
    count=0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):

            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                    count += 1
                except ValueError:
                    print(s[0])
                    continue

    print("{} of {} initialized".format(count, len(word_indices)))
    save_pickle("wemb", emb)
    return emb


def biLSTM(inputs, dim, seq_len, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Output of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd,
                                                                     inputs=inputs, dtype=tf.float32, scope=name)
    return hidden_states, cell_states

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def highway_layer(x, size, activation, name, carry_bias=-1.0):
    with tf.name_scope(name):
        W, b = weight_bias([size, size], [size])

        with tf.name_scope('transform_gate'):
            W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

        H = activation(tf.matmul(x, W) + b, name='activation')
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = tf.sub(1.0, T, name="carry_gate")

        y = tf.add(tf.mul(H, T), tf.mul(x, C), name='y') # y = (H * T) + (x * C)
        return y

def attention(x, name):
    batch, max_seq, dim = x.shape
    with tf.name_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        E = tf.nn.relu(tf.matmul(x, W) + b) # [batch, seq, dim]
        E2 = tf.matmul(E, E, transpose_b=True) # [ batch, seq, seq]
        a_raw = tf.matmul(E2, x) # [ batch, seq, dim]
        a_z = tf.reshape(tf.tile(tf.reduce_sum(E2, axis=2), dim), [batch, max_seq, dim]) #[batch, seq]
        a = tf.div(a_raw, a_z) #[batch, seq, dim]
        return a


