import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from BILSTM.common import *

def print_shape(name, tensor):
    print("{} : {}".format(name, tensor.shape))

class FAIRModel:

    def __init__(self, max_sequence, word_indice, batch_size, num_classes, vocab_size, embedding_size, lstm_dim):
        print("Building Model")

        self.input_p = tf.placeholder(tf.int32, [None, max_sequence], name="input_p")
        self.input_h = tf.placeholder(tf.int32, [None, max_sequence], name="input_h")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.filters = []
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_dim = lstm_dim
        self.W = None
        self.word_indice = word_indice
        self.network()
        self.sess = tf.session()


    def encode(self, s):
        embedded = tf.nn.embedding_lookup(self.embedding, s)
        hidden_states, cell_states = biLSTM(embedded, self.lstm_dim)

        h, size, _ = hidden_states
        print_shape("h:", h)
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, size, 1, 1],
            strides=[1, 1, 1, 1])

        print_shape("pooled:", pooled)
        return pooled

    def network(self):
        self.input_p, self.input_h, self.input_y = self.batch_input
        with tf.name_scope("embedding"), tf.device('/device:GPU:0'):
            self.embedding = load_embedding(self.word_indice, self.embedding_size)

        p_encode = self.encode(self.input_p)
        h_encode = self.encode(self.input_h)

        # TODO output
        f_concat = tf.concat([p_encode, h_encode])
        f_sub = p_encode - h_encode
        f_odot = tf.multiply(p_encode, h_encode)

        feature = tf.concat(f_concat, f_sub, f_odot)

        self.logits = tf.contrib.layers.fully_connected(feature, self.num_classes, activation_fn=tf.nn.relu)
        self.total_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.input_y,tf.int64)), tf.float32))

    def get_batches(self):
        ""

    def train(self, epochs, data):
        step_size = int((len(data)+epochs-1) / epochs)
        training_dataset = tf.data.Dataset.from_generator(self.get_batch())

        for i in range(epochs):
            batches = self.get_batches()
            for batch in batches:
                p, h, y = batch
                _, acc, loss = self.sess.run([self.train_op, self.acc, self.loss], feed_dict={
                    self.input_p: p,
                    self.input_h: h,
                    self.input_y: y
                })
                print("loss : {}".format(loss))




