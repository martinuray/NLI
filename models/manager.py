from tensorflow.python.client import device_lib

from models.CAFE import *
from util import *

LOCAL_DEVICES = device_lib.list_local_devices()
from tensorflow.python.client import timeline
from deepexplain.tensorflow import DeepExplain

def get_summary_path(name):
    i = 0
    def gen_path():
        return os.path.join('summary', '{}{}'.format(name, i))

    while os.path.exists(gen_path()):
        i += 1

    return gen_path()


class Manager:
    def __init__(self, max_sequence, word_indice, batch_size, num_classes, vocab_size, embedding_size, lstm_dim):
        print("Building Model")
        print("Batch size : {}".format(batch_size))
        print("LSTM dimension: {}".format(lstm_dim))
        self.input_p = tf.placeholder(tf.int32, [None, max_sequence], name="input_p_absolute_input")
        self.input_p_len = tf.placeholder(tf.int64, [None,], name="input_p_len_absolute_input")
        self.input_h = tf.placeholder(tf.int32, [None, max_sequence], name="input_h_absolute_input")
        self.input_h_len = tf.placeholder(tf.int64, [None,], name="input_h_len_absolute_input")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob_absolute_input")
        self.batch_size = batch_size
        self.filters = []

        self.num_classes = num_classes
        self.hidden_size = 200
        self.vocab_size = vocab_size
        self.sent_crop_len = 100

        self.embedding_size = embedding_size
        self.max_seq = max_sequence
        self.lstm_dim = lstm_dim
        self.reg_constant = 1e-6
        self.lr = 3e-4

        self.W = None
        self.word_indice = word_indice

        self.l2_loss = 0

        self.train_op = None
        config = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.device('/gpu:0'):
            self.network()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.get_train_op()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def log_info(self):
        self.run_metadata = tf.RunMetadata()
        path = get_summary_path("train")
        train_log_path = os.path.join(path, "train")
        test_log_path = os.path.join(path, "test")
        self.train_writer = tf.summary.FileWriter(train_log_path, self.sess.graph)
        self.train_writer.add_run_metadata(self.run_metadata, "train")
        print("Summary at {}".format(path))
        self.test_writer = tf.summary.FileWriter(test_log_path, filename_suffix=".test")

    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        return optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def network(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.Variable(load_pickle("wemb"), trainable=False)
        logits = cafe_network (self.input_p,
                               self.input_h,
                               self.input_p_len,
                               self.input_h_len,
                               self.batch_size,
                               self.num_classes,
                               self.embedding,
                               self.embedding_size,
                               self.max_seq,
                               self.l2_loss,
                               self.dropout_keep_prob
                               )
        self.logits = tf.identity(logits, name="absolute_output")
        print(self.input_y)
        pred_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1),tf.cast(self.input_y, tf.int64)), tf.float32))
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = pred_loss + self.reg_constant * l2_loss

        tf.summary.scalar('l2loss', self.reg_constant * l2_loss)
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.loss)

        return self.loss

    def load(self, id):
        self.saver.restore(self.sess, self.load_path(id))

    def save_path(self):
        return os.path.join(os.path.abspath("checkpoint"), "model")

    def load_path(self, id):
        return os.path.join(os.path.abspath("checkpoint"), id)

    def view_lrp(self, dev_data):
        def expand_y(y):
            r = []
            for yi in y:
                if yi == 0:
                    yp = [1,0,0]
                elif yi == 1:
                    yp = [0,1,0]
                else:
                    yp = [0,0,1]

                r.append(yp)
            return np.array(r)

        print("view lrp")
        dev_batches = get_batches(dev_data, 10, 100)
        p, p_len, h, h_len, y = dev_batches[0]

        p_emb = tf.get_variable("premise")
        with DeepExplain(session=self.sess) as de:
            logits = self.logits
            x_input = [self.input_p, self.input_p_len, self.input_h_len, self.input_h]
            xi = [p, p_len, h, h_len]
            yi = expand_y(y)
            print("grad : {}".format(tf.gradients(logits, p_emb)))
            E = de.explain('elrp', logits, p_emb, p)
            print("result----------")
            print(E)

    def print_time(self):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

    def check_dev(self, batches, g_step):
        acc_sum = []
        loss_sum = []
        step = 0
        for batch in batches:
            p, p_len, h, h_len, y = batch
            acc, loss, summary = self.sess.run([self.acc, self.loss, self.merged], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            }, run_metadata=self.run_metadata)
            acc_sum.append(acc)
            loss_sum.append(loss)
            self.test_writer.add_summary(summary, g_step+step)
            step += 1

        print("Dev acc={} loss={} ".format(avg(acc_sum), avg(loss_sum)))

    def train(self, epochs, data, valid_data):
        print("Train")
        self.log_info()
        self.sess.run(tf.global_variables_initializer())
        batches = get_batches(data, self.batch_size, self.sent_crop_len)
        dev_batches = get_batches(valid_data, 200, self.sent_crop_len)
        step_per_batch = int(len(data) / self.batch_size)
        log_every = int(step_per_batch/200)
        check_dev_every = int(step_per_batch/5)
        g_step = 0

        for i in range(epochs):
            print("Epoch {}".format(i))
            s_loss = 0
            l_acc = []
            time_estimator = TimeEstimator(len(batches), name="epoch")
            for batch in batches:
                g_step += 1
                p, p_len, h, h_len, y = batch
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                _, acc, loss, summary = self.sess.run([self.train_op, self.acc, self.loss, self.merged], feed_dict={
                    self.input_p: p,
                    self.input_p_len: p_len,
                    self.input_h: h,
                    self.input_h_len: h_len,
                    self.input_y: y,
                    self.dropout_keep_prob: 0.8,
                }, run_metadata=self.run_metadata, options=run_options)

                if g_step % log_every == 1 or g_step < 5:
                    print("step{} : {} acc : {} ".format(g_step, loss, acc))
                if g_step % check_dev_every == 0 :
                    self.check_dev(dev_batches, g_step)
                s_loss += loss
                l_acc.append(acc)
                self.train_writer.add_summary(summary, g_step)
                self.train_writer.add_run_metadata(self.run_metadata, "meta_{}".format(g_step))
                time_estimator.tick()

            current_step = tf.train.global_step(self.sess, self.global_step)
            path = self.saver.save(self.sess, self.save_path(), global_step=current_step)
            print("Checkpoint saved at {}".format(path))
            print("Training Average loss : {} , acc : {}".format(s_loss, avg(l_acc)))
