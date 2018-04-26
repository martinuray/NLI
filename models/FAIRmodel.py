import os
import time

from models.common import *

from tensorflow.python.client import device_lib
LOCAL_DEVICES = device_lib.list_local_devices()
#print("Viewable device : {}".format(LOCAL_DEVICES))


def get_summary_path(name):
    i = 0
    def gen_path():
        return os.path.join('summary', '{}{}'.format(name, i))

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
        self.reg_constant = 1e-5

        self.train_op = None

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        with tf.device('/gpu:0'):
            self.network()
            self.run_metadata = tf.RunMetadata()
            path = get_summary_path("train")
            self.train_writer = tf.summary.FileWriter(path, self.sess.graph)
            self.train_writer.add_run_metadata(self.run_metadata, "train")
            print("Summary at {}".format(path))
            self.test_writer = tf.summary.FileWriter(get_summary_path("test"), self.sess.graph)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


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
            self.embedding = load_embedding(self.word_indice, self.embedding_size)
            self.embedding = tf.Variable(load_pickle("wemb"))

        def encode(sent, name):
            hidden_states, cell_states = biLSTM(sent, self.lstm_dim, name)

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

        self.premise = tf.nn.embedding_lookup(self.embedding, self.input_p)
        self.hypothesis = tf.nn.embedding_lookup(self.embedding, self.input_h)

        p_encode = encode(self.premise, "premise")
        h_encode = encode(self.hypothesis, "hypothesis")

        f_concat = tf.concat([p_encode, h_encode], axis=1)
        f_sub = p_encode - h_encode
        f_odot = tf.multiply(p_encode, h_encode)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1)
        print_shape("feature:", feature)
        tf.summary.histogram("feature", feature)
        self.l2_loss = 0

        feature_drop = tf.nn.dropout(feature, self.dropout_keep_prob)
        hidden1 = self.dense(feature_drop, self.lstm_dim * 8, self.hidden_size, "hidden1")
        self.hidden1_drop = tf.nn.dropout(hidden1, self.dropout_keep_prob)
        hidden2 = self.dense(self.hidden1_drop, self.hidden_size, self.hidden_size, "hidden2")
        self.hidden2_drop = tf.nn.dropout(hidden2, self.dropout_keep_prob)

        logits = self.dense(self.hidden2_drop, self.hidden_size, self.num_classes, "output")
        return logits



    def get_sa(self):
        gradient = []
        for label in range(self.num_classes):
            g = tf.gradients(self.logits[:,label], [self.premise, self.hypothesis]) # [batch, max_seq, word]
            g_sum = tf.reduce_sum(tf.square(g), axis=3)
            print(g_sum)
            gradient.append(g_sum)
        return gradient



    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        return optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


    def network(self):
        logits = self.bilstm_network()
        self.logits = logits
        tf.summary.scalar('l2loss', self.reg_constant * self.l2_loss)
        print_shape("self.logits:", self.logits)
        pred_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1),tf.cast(self.input_y, tf.int64)), tf.float32))
        self.loss = pred_loss + self.reg_constant * self.l2_loss

        tf.summary.histogram('logit_histogram', self.logits)
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

        self.SA = self.get_sa()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.get_train_op()

        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()

    def get_batches(self, data, batch_size):
        # data is fully numpy array here
        step_size = int((len(data) + batch_size - 1) / batch_size)
        new_data = []
        for step in range(step_size):
            p = []
            p_len = []
            h = []
            h_len = []
            y = []
            for i in range(batch_size):
                idx = step * batch_size + i
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

    def check_dev(self, batches):
        acc_sum = []
        loss_sum = []
        for batch in batches:
            p, p_len, h, h_len, y = batch
            acc, loss, summary, step = self.sess.run([self.acc, self.loss, self.merged, self.global_step], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            }, run_metadata=self.run_metadata)
            self.test_writer.add_summary(summary, global_step=step)
            acc_sum.append(acc)
            loss_sum.append(loss)

        print("Dev acc={} loss={} ".format(avg(acc_sum), avg(loss_sum)))

    def sa_analysis(self, data, idx2word):
        batches = self.get_batches(data, 50)
        D = {0: "E",
             1: "N",
             2: "C"}

        def word(index):
            if index in idx2word:
                return idx2word[index]
            else:
                return "OOV"
        for batch in batches:
            p, p_len, h, h_len, y = batch
            acc, logits, sa = self.sess.run([self.acc, self.logits, self.SA], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            l, = y.shape
            pred = np.argmax(logits, axis=1)
            for i in range(l):
                true_label = D[y[i]]
                pred_label = D[pred[i]]
                print("--- {}({}) --- ".format(pred_label, true_label))
                for title, sent, s_len, s_idx in [("prem", p, p_len ,0), ("hypo", h, h_len ,1)]:
                    print("{}:".format(title), end="\t")
                    for j in range(s_len[i]+10):
                        print("{}".format(word(sent[i, j])), end=" ")
                    print("")
                    for label in range(self.num_classes):
                        total_d = np.sum(sa[label][s_idx, i, :])
                        print("{0}{1:.1f} : ".format(D[label], total_d), end="\t")
                        for j in range(s_len[i]+10):
                            char_len = len(word(sent[i,j]))
                            buffer = " " * (char_len-4)
                            print("{0:.2f}{1}".format(sa[label][s_idx, i, j], buffer), end=" ")
                        print("")

    def load(self, id):
        self.saver.restore(self.sess, self.load_path(id))

    def save_path(self):
        directory = os.path.join(os.path.abspath("checkpoint"), "model")
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def load_path(self, id):
        return os.path.join(os.path.abspath("checkpoint"), id)

    def train(self, epochs, data, valid_data):
        print("Train")
        self.sess.run(tf.global_variables_initializer())
        batches = self.get_batches(data, self.batch_size)
        dev_batches = self.get_batches(valid_data, 200)
        check_dev_every = 2000
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


                if g_step % 100 == 1:
                    print("step{} : {} acc : {} time={}".format(g_step, loss, acc, time.time()))
                if g_step % check_dev_every == 0 :
                    self.check_dev(dev_batches)
                s_loss += loss
                l_acc.append(acc)
                self.train_writer.add_summary(summary, g_step)
                self.train_writer.add_run_metadata(self.run_metadata, "meta_{}".format(g_step))

            current_step = tf.train.global_step(self.sess, self.global_step)
            path = self.saver.save(self.sess, self.save_path(), global_step=current_step)
            print("Checkpoint saved at {}".format(path))
            print("Average loss : {} , acc : {}".format(s_loss, avg(l_acc)))

    def view_weights(self, data):
        batches = self.get_batches(data, 50)
        D = {0: "E",
             1: "N",
             2: "C"}
        o_w = tf.get_default_graph().get_tensor_by_name("output/W:0")
        p, p_len, h, h_len, y = batches[0]
        out_o_w, = self.sess.run([o_w], feed_dict={
            self.input_p: p,
            self.input_p_len: p_len,
            self.input_h: h,
            self.input_h_len: h_len,
            self.input_y: y,
            self.dropout_keep_prob: 1.0,
        })

        for i in range(200):
            ce = out_o_w[i,2] - out_o_w[i,0]
            en = out_o_w[i,0] - out_o_w[i,1]
            print("{0:.2f} {1:.2f}".format(ce, en))
