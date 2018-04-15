
import numpy as np
import tensorflow as tf


def dense(input, input_size, output_size, name):
    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=[input_size, output_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[output_size]), name="b")
        l2_loss = tf.nn.l2_loss(W)
        output = tf.nn.xw_plus_b(input, W, b, name="logits")  # [batch, num_class]

        return output, l2_loss, W, b

import random


rx = []
ry = []

for i in range(50):
    a = random.randint(1,9)
    b = random.randint(2,8)
    y = random.randint(0,1)
    c = 12 - y * 10 + random.randint(0,2)
    d  = 10
    ry.append(y)
    rx.append([a,b,c,d])



X0 = np.array([[4,2,3,10], [4,3,12,10], [7,8,2,10]], dtype=np.float32)
Y0 = np.array([1, 0, 1])

X = np.array(rx, dtype=np.float32)
Y = np.array(ry, dtype=np.float32)

input = tf.placeholder(tf.float32, [None, 4], name="input")
label = tf.placeholder(tf.int32, [None], name="input")
logits, l2_loss, W, b = dense(input, 4, 2, "layer")

pred_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
loss = pred_loss + l2_loss * 0.01

optimizer = tf.train.AdamOptimizer(2e-3)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

SA_scores = [tf.square(tf.gradients(logits[:,i], input)) for i in range(2)]

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                             log_device_placement=False))
sess.run(tf.global_variables_initializer())

for epoch in range(2000):
    _, loss_out, logits_out = sess.run([train_op, loss, logits], feed_dict={
        input: X,
        label: Y,
    })
    print("Loss : {}".format(loss_out))

for i in range(3):
    print("{} {}".format(X[i], Y[i]))

W_out, b_out = sess.run([W, b], feed_dict={
        input: X0[i:i+1,:],
        label: Y0[i:i+1],
    })
print("W:{} B:{}".format(W_out, b_out))

for i in range(3):
    g, l = sess.run([SA_scores, logits], feed_dict={
        input: X0[i:i+1,:],
        label: Y0[i:i+1],
    })
    print("label {} grad : {}".format(np.argmax(l, axis=1), g))



