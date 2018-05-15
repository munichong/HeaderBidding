import numpy as np, pickle
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.DataReader import SurvivalData
from survival_analysis.Distributions import WeibullDistribution

class ParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k=1, learning_rate=0.001):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate


    def left_censoring(self, dist, time, Lambda):
        return dist.gradient(time, Lambda) - dist.gradient(tf.constant(0.0), Lambda)

    def right_censoring(self, dist, time, Lambda):
        return dist.gradient(tf.constant(np.inf), Lambda) - dist.gradient(time, Lambda)

    def linear_regression(self, predictors, weights):
        feat_vals = tf.tile(tf.expand_dims(predictors, axis=-1), [1, 1, 1])
        feat_x_weights = tf.reduce_sum(tf.multiply(weights, feat_vals), axis=1)
        intercept = tf.Variable(tf.constant(0.1))
        return feat_x_weights + intercept

    def factorization_machines(self):
        pass

    def run_graph(self, num_features, train_data, val_data, test_data):
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

        time = tf.placeholder(tf.float32, shape=[None], name='time')
        event = tf.placeholder(tf.int32, shape=[None], name='event')

        embeddings = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=0.5))


        if self.k == 1:
            ''' treat the input_vectors as masks '''
            ''' input_vectors do NOT need to be binary vectors '''
            Lambda = self.linear_regression(input_vectors, embeddings)

        else:
            Lambda = self.factorization_machines()


        ''' 
        if event == 0, right-censoring
        if event == 1, left-censoring 
        '''
        survival = tf.where(tf.equal(event, 1),
                           self.left_censoring(self.distribution, time, Lambda),
                           self.right_censoring(self.distribution, time, Lambda))

        not_survival = 1 - survival

        logloss = tf.losses.log_loss(labels=event, predictions=not_survival, weights=1.0)
        loss_mean = tf.reduce_mean(logloss)
        training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_mean)


        not_survival_binary = tf.where(tf.greater_equal(not_survival, 0.5),
                                       tf.ones(tf.shape(not_survival)),
                                       tf.zeros(tf.shape(not_survival)))
        auc, auc_update = tf.metrics.auc(labels=event, predictions=not_survival_binary, weights=None)
        acc, acc_update = tf.metrics.accuracy(labels=event, predictions=not_survival_binary, weights=None)

        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


        with tf.Session() as sess:
            init.run()

            num_total_batches = int(np.ceil(train_data.num_instances / self.batch_size))
            next_train_batch = train_data.make_batch(self.batch_size)
            for epoch in range(1, self.num_epochs + 1):
                # model training
                num_batch = 0
                for duration_batch, event_batch, features_batch in next_train_batch:
                    num_batch += 1
                    sess.run(running_vars_initializer)
                    _, loss_batch, _, _, survival_batch = sess.run([training_op, loss_mean, auc_update, acc_update, survival],
                                                                   feed_dict={input_vectors: features_batch,
                                                                              time: duration_batch,
                                                                              event: event_batch})

                    auc_batch, acc_batch = sess.run([auc, acc])
                    # print(loss_batch)
                    # print(auc_batch)
                    # print(acc_batch)
                    print(survival_batch)
                    print("Epoch %d - Batch %d/%d: loss = %.4f, auc = %.4f, accuracy = %.4f" %
                          (epoch, num_batch, num_total_batches, loss_batch, auc_batch, acc_batch))



                # evaluation on training data
                eval_nodes_batch = [loss_mean, auc_update, acc_update, survival]
                eval_nodes_all = [auc, acc, survival]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, auc_train, acc_train = self.evaluate(train_data, sess, eval_nodes_batch, eval_nodes_all)
                print("*** On Training Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_train, auc_train, acc_train))

                # evaluation on validation data
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_val, auc_val, acc_val = self.evaluate(val_data, sess, eval_nodes_batch, eval_nodes_all)
                print("*** On Validation Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_val, auc_val, acc_val))

                # evaluation on test data
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_test, auc_test, acc_test = self.evaluate(test_data, sess, eval_nodes_batch, eval_nodes_all)
                print("*** On Test Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_test, auc_test, acc_test))



    def evaluate(self, next_batch, sess, eval_nodes_batch, eval_nodes_all):
        for duration_batch, event_batch, features_batch in sess.run(next_batch):
            _ = sess.run(eval_nodes_batch,
                                                        feeddict={'input_vectors:0': features_batch,
                                                                  'time:0': duration_batch,
                                                                  'event:0': event_batch})

        return sess.run(eval_nodes_all)


if __name__ == "__main__":
    with open('../Vectors_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(distribution = WeibullDistribution(),
                    batch_size = 512,
                    num_epochs = 30)
    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('../Vectors_train.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_val.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_test.p', 'rb'))),
                    )
