import numpy as np
from parameter import *

import tensorflow as tf


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

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):

            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                except ValueError:
                    print(s[0])
                    continue

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

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states

