import random
import time
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

def get_summary_path():
    i = 0
    def gen_path():
        return os.path.join('summary', 'train{}'.format(i))

    while os.path.exists(gen_path()):
        i += 1

    return gen_path()

class FAIRModel:

    def __init__(self, max_sequence, word_indice, batch_size, num_classes, vocab_size, embedding_size, lstm_dim):
        print("Building Model")
        print("Batch size : {}".format(batch_size))
        print("LSTM dimension: {}".format(lstm_dim))
        self.input_p = tf.placeholder(tf.int32, [None, max_sequence], name="input_p")
        self.input_p_len = tf.placeholder(tf.int64, [None,], name="input_p_len")
        self.input_h = tf.placeholder(tf.int32, [None, max_sequence], name="input_h")
        self.input_h_len = tf.placeholder(tf.int64, [None,], name="input_h_len")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.filters = []

        self.num_classes = num_classes
        self.hidden_size = 200
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq = max_sequence
        self.lstm_dim = lstm_dim
        self.W = None
        self.word_indice = word_indice
        self.reg_constant = 1e-6

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        with tf.device('/gpu:0'):
            self.bilstm_network()
            self.run_metadata = tf.RunMetadata()
            path = get_summary_path()
            self.train_writer = tf.summary.FileWriter(path, self.sess.graph)
            self.train_writer.add_run_metadata(self.run_metadata, "train")
            print("Summary at {}".format(path))
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


    def dense(self, input, input_size, output_size, name):
        with tf.variable_scope(name):
            W = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.01, shape=[output_size]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            #self.l2_loss += tf.nn.l2_loss(b)
            output = tf.nn.xw_plus_b(input, W, b, name="logits")  # [batch, num_class]
            return output

    def bilstm_network(self):
        print("network..")
        with tf.name_scope("embedding"):
            #self.embedding = load_embedding(self.word_indice, self.embedding_size)
            self.embedding = tf.Variable(load_pickle("wemb"))

        def encode(self, s, s_len, embedding, name):
            embedded = tf.nn.embedding_lookup(embedding, s)
            tf.summary.histogram(name, embedded[0, :])
            hidden_states, cell_states = biLSTM(embedded, self.lstm_dim, s_len, name)

            fw, bw = hidden_states
            all_enc = tf.concat([fw, bw], axis=2)

            pooled = tf.nn.max_pool(
                tf.reshape(all_enc, [-1, self.max_seq, 2 * self.lstm_dim, 1]),
                ksize=[1, self.max_seq, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )

            print_shape("pooled:", pooled)
            return tf.reshape(pooled, [-1, 2 * self.lstm_dim])

        #print("self.embedding : {}".format(self.embedding))
        p_encode = encode(self.input_p, self.input_p_len, "premise")
        h_encode = encode(self.input_h, self.input_h_len, "hypothesis")

        f_concat = tf.concat([p_encode, h_encode], axis=1)
        f_sub = p_encode - h_encode
        f_odot = tf.multiply(p_encode, h_encode)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1)
        print_shape("feature:", feature)
        tf.summary.histogram("feature", feature)
        l2_loss = 0

        feature_drop = tf.nn.dropout(feature, self.dropout_keep_prob)
        hidden1 = self.dense(feature_drop, self.lstm_dim * 8, self.hidden_size, "hidden1")
        self.hidden1_drop = tf.nn.dropout(hidden1, self.dropout_keep_prob)
        hidden2 = self.dense(self.hidden1_drop, self.hidden_size, self.hidden_size, "hidden2")
        self.hidden2_drop = tf.nn.dropout(hidden2, self.dropout_keep_prob)
        logits = self.dense(self.hidden2_drop, self.hidden_size, self.num_classes, "output")
        return logits, l2_loss

    def cafe_network(self):
        print("CAFE network..")
        self.highway_size = 300
        with tf.name_scope("embedding"):
            #self.embedding = load_embedding(self.word_indice, self.embedding_size)
            self.embedding = tf.Variable(load_pickle("wemb"), trainable=False)

        def encode(sent, name):
            embedded_raw = tf.nn.embedding_lookup(self.embedding, sent)  # [batch, max_seq, dim]

            embedded = tf.reshape(embedded_raw, [-1, self.embedding_size])
            h = highway_layer(embedded, self.highway_size, tf.nn.relu, "{}/high1".format(name))
            h2 = highway_layer(h, self.highway_size, tf.nn.relu, "{}/high2".format(name))
            h_out = tf.reshape(h2, [self.batch_size, self.max_seq, self.embedding_size])
            att = attention(h_out, name)

            return h_out, att

        p_encode = encode(self.input_p, "premise")
        h_encode = encode(self.input_h, "hypothesis")

        f_concat = tf.concat([p_encode, h_encode], axis=1)
        f_sub = p_encode - h_encode
        f_odot = tf.multiply(p_encode, h_encode)





    def network(self):
        logits, l2_loss = self.bilstm_network()
        self.logits = logits
        tf.summary.scalar('l2loss', self.reg_constant * l2_loss)
        print_shape("self.logits:", self.logits)
        pred_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1),tf.cast(self.input_y, tf.int64)), tf.float32))
        self.loss = pred_loss + self.reg_constant * l2_loss

        tf.summary.histogram('logit_histogram', self.logits)
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()

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
                start = time.time()
                p, p_len, h, h_len, y = batch
                _, acc, loss, summary = self.sess.run([self.train_op, self.acc, self.loss, self.merged], feed_dict={
                    self.input_p: p,
                    self.input_p_len: p_len,
                    self.input_h: h,
                    self.input_h_len: h_len,
                    self.input_y: y,
                    self.dropout_keep_prob: 0.9,
                }, run_metadata=self.run_metadata)

                elapsed = time.time() - start
                print("step{} : {} acc : {} elapsed={}".format(g_step, loss, acc, elapsed))
                s_loss += loss
                l_acc.append(acc)
                self.train_writer.add_summary(summary, g_step)
                self.train_writer.add_run_metadata(self.run_metadata, "meta_{}".format(g_step))
            print("Average loss : {} , acc : {}".format(s_loss, avg(l_acc)))
            current_step = tf.train.global_step(self.sess, self.global_step)
            path = self.saver.save(self.sess, os.paht.abspath("checkpoint"), global_step=current_step)
            print("Checkpoint saved at {}".format(path))
