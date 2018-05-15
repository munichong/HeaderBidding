import tensorflow as tf
import numpy as np

def func_for_event1(a, t):
    return tf.gradients(t + a, [t])[0]

def func_for_event0(a, t):
    return tf.zeros_like(t)

time = tf.placeholder(tf.float32, shape=[None])  # [3.2, 4.2, 1.0, 1.05, 1.8]
event = tf.placeholder(tf.int32, shape=[None])  # [0, 1, 1, 0, 1]

# result: [2.2, 5.2, 2.0, 0.05, 2.8]
result = tf.where( tf.equal( 1, event ), func_for_event1( 1, time ), func_for_event0( 1, time ) )
# result: [2.2, 5.2, 2.0, 0.05, 2.8]
# For example, 3.2 should be sent to func_for_event0 because the first element in event is 0.

not_survival_binary = tf.where(tf.greater_equal(time, 1.5), tf.ones(tf.shape(time)), tf.zeros(tf.shape(time)))
accuracy, acc_op = tf.metrics.accuracy(labels=event, predictions=not_survival_binary, weights=None)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    init.run()
    for e in [np.array([1, 0, 0, 0, 1]), np.array([1, 1, 1, 1, 1])]:
        res = sess.run( acc_op, feed_dict = {
            time : np.array( [3.2, 4.2, 1.0, 1.05, 1.8] ),
            event: e
        } )
        print ( sess.run(accuracy) )
    print(sess.run(accuracy))
