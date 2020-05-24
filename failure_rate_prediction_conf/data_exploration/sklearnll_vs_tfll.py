import tensorflow as tf
from sklearn.metrics import log_loss

y_true = [1, 1, 1, 0, 0]
y_pred = [0.5, 0.9, 0.8, 0.1, 0.1]

tf_log_loss = tf.losses.log_loss(predictions=tf.constant(y_pred), labels=tf.constant(y_true))

with tf.Session() as sess:
    a = sess.run(tf_log_loss)
    print("sklearn log-loss:", log_loss(y_true, y_pred))
    print("tensorflow log-loss:", a)
