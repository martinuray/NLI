import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from BILSTM.common import *

from tensorflow.python.client import device_lib
LOCAL_DEVICES = device_lib.list_local_devices()
#print("Viewable device : {}".format(LOCAL_DEVICES))


def print_shape(name, tensor):
    print("{} : {}".format(name, tensor.shape))

class FAIRModel:

    def __init__(self, max_sequence, word_indice, batch_size, num_classes, vocab_size, embedding_size, lstm_dim):
        print("Building Model")

        self.input_p = tf.placeholder(tf.int32, [None, max_sequence], name="input_p")
        self.input_p_len = tf.placeholder(tf.int64, [None,], name="input_p_len")
        self.input_h = tf.placeholder(tf.int32, [None, max_sequence], name="input_h")
        self.input_h_len = tf.placeholder(tf.int64, [None,], name="input_h_len")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.filters = []

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq = max_sequence
        self.lstm_dim = lstm_dim
        self.W = None
        self.word_indice = word_indice
        with tf.device('/gpu:0'):
            self.network()

        tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.sess = tf.Session()


    def encode(self, s, s_len, name):
        embedded = tf.nn.embedding_lookup(self.embedding, s)
        hidden_states, cell_states = biLSTM(embedded, self.lstm_dim, s_len, name)

        fw, bw = hidden_states
        all_enc = tf.concat([fw, bw], axis=2)

        pooled = tf.nn.max_pool(
            tf.reshape(all_enc, [-1, self.max_seq, 2*self.lstm_dim, 1]),
            ksize=[1, self.max_seq, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
        )

        print_shape("pooled:", pooled)
        return tf.reshape(pooled, [-1, 2*self.lstm_dim])

    def network(self):
        print("network..")
        with tf.name_scope("embedding"):
            self.embedding = load_embedding(self.word_indice, self.embedding_size)

        p_encode = self.encode(self.input_p, self.input_p_len, "premise")
        h_encode = self.encode(self.input_h, self.input_h_len, "hypothesis")

        f_concat = tf.concat([p_encode, h_encode], axis=1)
        f_sub = p_encode - h_encode
        f_odot = tf.multiply(p_encode, h_encode)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1)
        print_shape("feature:", feature)

        self.logits = tf.contrib.layers.fully_connected(feature, self.num_classes, activation_fn=tf.nn.relu)
        print_shape("self.logits:", self.logits)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1),tf.cast(self.input_y, tf.int64)), tf.float32))

        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def get_batches(self, data):
        # data is fully numpy array here
        step_size = int((len(data) + self.batch_size - 1) / self.batch_size)
        new_data = []
        for step in range(step_size):
            p = []
            p_len = []
            h = []
            h_len = []
            y = []
            for i in range(self.batch_size):
                idx = step * self.batch_size + i
                if idx >= len(data):
                    break
                p.append(data[idx]['p'])
                p_len.append(data[idx]['p_len'])
                h.append(data[idx]['h'])
                h_len.append(data[idx]['h_len'])
                y.append(data[idx]['y'])
            if len(p) > 0:
                new_data.append((np.stack(p), np.stack(p_len), np.stack(h), np.stack(h_len), np.stack(y)))
        print("Total of {} batches".format(len(new_data)))
        return new_data

    def train(self, epochs, data):
        print("Train")
        self.sess.run(tf.global_variables_initializer())
        batches = self.get_batches(data)
        g_step= 0
        for i in range(epochs):
            print("Epoch {}".format(i))
            s_loss = 0
            l_acc = []
            for batch in batches:
                g_step += 1
                p, p_len, h, h_len, y = batch
                _, acc, loss = self.sess.run([self.train_op, self.acc, self.loss], feed_dict={
                    self.input_p: p,
                    self.input_p_len: p_len,
                    self.input_h: h,
                    self.input_h_len: h_len,
                    self.input_y: y,
                    self.dropout_keep_prob: 0.5,
                })
                print("step{} : {} acc : {}".format(g_step, loss, acc))
                s_loss += loss
                l_acc.append(acc)
            print("Average loss : {} , acc : {}".format(s_loss, avg(l_acc)))
