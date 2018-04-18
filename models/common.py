import pickle

import numpy as np
import tensorflow as tf

from parameter import *


def avg(l):
    return sum(l)/len(l)


def load_pickle(name):
    path = os.path.join("pickle", name)
    return pickle.load(open(path,"rb"))


def save_pickle(name, obj):
    path = os.path.join("pickle", name)
    return pickle.dump(obj, open(path,"wb"))

def reverse_index(word2idx):
    idx2word = dict()
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word

def print_shape(name, tensor):
    print("{} : {}".format(name, tensor.shape))


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


def get_batches(data, batch_size, crop_max = 100):
    # data is fully numpy array here
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        p = []
        p_len_list = []
        h = []
        h_len_list = []
        y = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            p_len = min(data[idx]['p_len'], crop_max)
            p.append(data[idx]['p'][:crop_max])
            p_len_list.append(p_len)

            h_len = min(data[idx]['h_len'], crop_max)
            h.append(data[idx]['h'][:crop_max])
            h_len_list.append(h_len)

            y.append(data[idx]['y'])
        if len(p) > 0:
            new_data.append((np.stack(p), np.stack(p_len_list), np.stack(h), np.stack(h_len_list), np.stack(y)))
    return new_data


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


def biLSTM(inputs, dim, name):
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
    W = tf.get_variable(
        "weight",
        shape=W_shape,
        regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def highway_layer(x, size, activation, name, carry_bias=-1.0):
    with tf.variable_scope(name):
        W, b = weight_bias([size, size], [size])

        with tf.variable_scope('transform_gate'):
            W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

        H = activation(tf.matmul(x, W) + b, name='activation')  # [batch, out_size]
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = 1.0 - T

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y') # y = (H * T) + (x * C)
        return y



def seq_xw_plus_b(x, W, b):
    input_shape = tf.shape(x)
    dim, out_dim = W.get_shape().as_list()
    print(x)
    x_flat = tf.reshape(x, shape=[-1, dim])
    z = tf.nn.xw_plus_b(x_flat, W, b)
    print_shape("input_shape[:2]", input_shape[:2])
    print_shape("tf.shape(W)[1:]", tf.shape(W)[1:])
    out_shape = tf.concat([input_shape[:2],tf.shape(W)[1:]], axis=0)
    print_shape("out_shape", out_shape)
    out = tf.reshape(z, shape=out_shape)
    print_shape("out", out)
    return out

def attention(x, name):
    x = tf.cast(x, dtype=tf.float32)
    _, max_seq, dim = x.get_shape().as_list()
    print(dim)
    with tf.variable_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        print_shape("x", x)
        E = tf.nn.relu(seq_xw_plus_b(x, W, b)) # [batch, seq, dim]
        print_shape("E", E)
        E2 = tf.matmul(E, E, transpose_b=True) # [ batch, seq, seq]
        print_shape("E2", E2)
        a_raw = tf.matmul(E2, x) # [ batch, seq, dim]
        a_z = tf.reshape(tf.tile(tf.reduce_sum(E2, axis=2), [1,dim]), tf.shape(x)) #[batch, seq]
        a = tf.div(a_raw, a_z) #[batch, seq, dim]
        return a

def interaction_feature(a,b, axis):
    f_concat = tf.concat([a, b], axis=axis)
    f_sub = a - b
    f_odot = tf.multiply(a, b)
    return f_concat, f_sub, f_odot


def dense(input, input_size, output_size, l2_loss, name):
    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=[input_size, output_size],
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[output_size]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        output = tf.nn.xw_plus_b(input, W, b)  # [batch, num_class]
        return output

def cartesian(v1, v2):
    # [d1] [d2], -> [d1,d2] [d2,d1]
    _, d1, = v1.get_shape().as_list()
    _, d2, = v2.get_shape().as_list()
    v1_e = tf.tile(v1, [1,d2]) # [batch*seq, d1*d2]
    v1_flat = tf.reshape(v1_e, [-1, d1, d2])
    v2_e = tf.tile(v2, [1,d1]) #
    v2_flat = tf.reshape(v2_e, [-1, d2, d1])
    print_shape("v1_flat", v1_flat)
    print_shape("v2_flat", v2_flat)
    return tf.matmul(v1_flat, v2_flat) # [batch*seq, d1, d1]


def factorization_machine(input, n_item, input_size, l2_loss, name):
    # input : [ -1, input_size]
    hidden_dim = 99
    with tf.variable_scope(name):
        L = tf.reshape(dense(input, input_size, 1, l2_loss, "w"), [-1]) # [batch*seq]

        v = tf.get_variable(
            "v",
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            shape=[input_size, hidden_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        v2 = tf.matmul(v, v, transpose_b=True) #[input_size, input_size)
        if False:
            x2 = cartesian(input, input) # [batch*seq, input_size , input_size]
            P_list = []
            pp = tf.nn.conv2d(input=tf.reshape(x2, [-1, input_size, input_size, 1]),
                         filter=tf.reshape(v2, [input_size,input_size,1,1]),
                         strides=[1,1,1,1],
                         padding="VALID"
                         )
            P = tf.reshape(pp, [-1])
        P1 = tf.pow(tf.matmul(input, v), 2)
        P2 = tf.matmul(tf.pow(input, 2), tf.pow(v, 2))
        P = tf.multiply(0.5,
            tf.reduce_sum(P1-P2,1))
        print_shape("L", L)
        print_shape("P", P)
        return L + P
"""
        P = tf.zeros(tf.shape(input)[:1])
        def t(arr):
            return tf.matrix_transpose(arr)
        for d in range(input_size):
            h1 = t(tf.multiply(t(input), input[:,d])) #[-, input_size
            h2 = tf.multiply(h1, v2[d,:])
            P = P + tf.reduce_sum(h2, axis=1)
            # TODO is it correct?
"""



def LSTM_pool(input, max_seq, input_len, dim, name):
    with tf.variable_scope(name):
        lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=dim)


    batch_max_len = tf.cast(tf.reduce_max(input_len), dtype=tf.int32)
    input_crop = input[:, :batch_max_len,:]
    #hidden_states, cell_states = tf.nn.dynamic_rnn(cell=lstm_fwd,
    #                                               inputs=input_crop,
    #                                               dtype=tf.float32,
    #                                               scope=name)
    hidden_states, cell_states = lstm_cell(input_crop, dtype=tf.float32, scope=name)
    # [batch, batch_max_len, dim]
    h_shape = tf.shape(hidden_states)
    mid = tf.constant(max_seq, shape=[1]) - h_shape[1]
    print(h_shape[0:1])
    print(mid)
    pad_size = tf.concat([h_shape[0:1], mid, h_shape[2:]], axis=0)
    print(pad_size)
    hidden_states = tf.concat([hidden_states, tf.zeros(pad_size)], axis=1)

    max_pooled = tf.nn.max_pool(
        tf.reshape(hidden_states, [-1, max_seq, dim, 1]),
        ksize=[1, max_seq, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    avg_pooled = tf.nn.avg_pool(
        tf.reshape(hidden_states, [-1, max_seq, dim, 1]),
        ksize=[1, max_seq, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    return tf.reshape(tf.concat([max_pooled, avg_pooled], axis=2),[-1, dim*2], name="encode_{}".format(name))



