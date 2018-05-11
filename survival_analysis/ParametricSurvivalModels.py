import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.TFDataReader import TFDataReader
from survival_analysis.Distributions import WeibullDistribution

class ParametricSurvival:

    def __init__(self, train_file_path, val_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

    # def __init__(self, training_data_reader):
    #     self.train_data_reader = training_data_reader
    #     self.val_data_reader = training_data_reader
    #     self.test_data_reader = training_data_reader

    def left_censoring(self, dist, time, Lambda):
        return dist.gradient(time, Lambda) - dist.gradient(0.0, Lambda)

    def right_censoring(self, dist, time, Lambda):
        return dist.gradient(np.inf, Lambda) - dist.gradient(time, Lambda)

    def linear_regression(self, predictors, weights):
        feat_vals = tf.tile(tf.expand_dims(predictors, axis=-1), [1, 1])
        feat_x_weights = tf.reduce_sum(tf.multiply(weights, feat_vals), axis=1)
        intercept = tf.Variable(tf.constant(0.1))
        return feat_x_weights + intercept

    def run_graph(self, distribution, num_features, batch_size, num_epochs, k=1, learning_rate=0.001):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 1
        :return:
        '''
        # INPUTs
        input_vectors = tf.placeholder(tf.float32,
                                  shape=[None, num_features],
                                  name='input_vectors')

        time = tf.placeholder(tf.float32, shape=(), name='time')
        event = tf.placeholder(tf.int32, shape=(), name='event')

        embeddings = tf.Variable(tf.truncated_normal(shape=(num_features, k), mean=0.0, stddev=0.5))

        if k == 1:
            ''' treat the input_vectors as masks '''
            ''' input_vectors do NOT need to be binary vectors '''
            Lambda = self.linear_regression(input_vectors, embeddings)

        else:
            Lambda = None
            pass

        ''' 
        if event == 0, right-censoring
        if event == 1, left-censoring 
        '''
        survival = tf.cond(event == 1,
                           lambda: self.left_censoring(distribution, time, Lambda),
                           lambda: self.right_censoring(distribution, time, Lambda))

        not_survival = 1 - survival

        logloss = tf.losses.log_loss(labels=event, predictions=not_survival, weights=1.0)
        loss_mean = tf.reduce_mean(logloss)
        training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_mean)

        auc = tf.metrics.auc(labels=event, predictions=not_survival, weights=None)
        not_survival_binary = tf.cond(not_survival>=0.5, lambda: 1, lambda: 0)
        accuracy = tf.metrics.accuracy(labels=event, predictions=not_survival_binary, weights=None)

        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            init.run()
            train_data_reader = TFDataReader(self.train_file_path)
            num_total_batches = int(np.ceil(len(train_data_reader.num_data) / batch_size))
            next_train_batch = train_data_reader.make_batch(batch_size)
            for epoch in range(1, num_epochs + 1):
                # model training
                num_batch = 0
                for duration_batch, event_batch, features_batch in sess.run(next_train_batch):
                    num_batch += 1
                    _, loss_batch, auc_batch, acc_batch = sess.run([training_op, loss_mean, auc, accuracy],
                                                                   feed_dict={input_vectors: features_batch,
                                                                              time: duration_batch,
                                                                              event:event_batch})
                    print("Epoch %d - Batch %d/%d: loss = %.4f, auc = %.4f, accuracy = %.4f" %
                          (epoch, num_batch, num_total_batches, loss_batch, auc_batch, acc_batch))

                # evaluation on training data
                eval_nodes = [not_survival, not_survival_binary]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, auc_train, acc_train = self.evaluate(TFDataReader(self.train_file_path).make_batch(batch_size),
                                                                 sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_train, auc_train, acc_train))

                # evaluation on validation data
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_val, auc_val, acc_val = self.evaluate(TFDataReader(self.val_file_path).make_batch(batch_size),
                                                               sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_val, auc_val, acc_val))

                # evaluation on test data
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_test, auc_test, acc_test = self.evaluate(TFDataReader(self.test_file_path).make_batch(batch_size),
                                                           sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_test, auc_test, acc_test))



    def evaluate(self, next_batch, sess, eval_nodes):
        not_survival_all = []
        not_survival_binary_all = []
        true_event_all = []
        num_batch = 0
        for duration_batch, event_batch, features_batch in sess.run(next_batch):
            num_batch += 1
            not_survival_batch, not_survival_binary_batch = sess.run(eval_nodes,
                                                        feeddict={'input_vectors:0': features_batch,
                                                                  'time:0': duration_batch,
                                                                  'event:0': event_batch})
            not_survival_all.extend(not_survival_batch)
            not_survival_binary_all.extend(not_survival_binary_batch)
            true_event_all.extend(event_batch)
        return log_loss(true_event_all, not_survival_all), \
               roc_auc_score(true_event_all, not_survival_all), \
               accuracy_score(true_event_all, not_survival_binary_all)


if __name__ == "__main__":
    with open('../Vectors_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(train_file_path='../Vectors_train.csv',
                               val_file_path='../Vectors_val.csv',
                               test_file_path='../Vectors_test.csv')
    model.run_graph(distribution = WeibullDistribution(),
                    num_features = num_features,
                    batch_size = 512,
                    num_epochs = 30)

