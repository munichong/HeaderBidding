import numpy as np, pickle

import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.DataReader import SurvivalData
from survival_analysis.Distributions import WeibullDistribution

class ParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k=1, learning_rate=0.01):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate


    def regression(self, predictors, weights):
        feat_vals = tf.tile(tf.expand_dims(predictors, axis=-1), [1, 1, 1])
        feat_x_weights = tf.reduce_sum(tf.multiply(weights, feat_vals), axis=1)
        intercept = tf.Variable(tf.constant(0.1))

        return tf.exp(tf.squeeze(feat_x_weights + intercept, [-1]))

    def factorization_machines(self):
        pass

    def run_graph(self, num_features, train_data, val_data, test_data, sample_weights=None):
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

        embeddings = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=0.02))


        if self.k == 1:
            ''' treat the input_vectors as masks '''
            ''' input_vectors do NOT need to be binary vectors '''
            Lambda = self.regression(input_vectors, embeddings)

        else:
            Lambda = self.factorization_machines()

        ''' 
        if event == 0, right-censoring
        if event == 1, left-censoring 
        '''
        survival = tf.where(tf.equal(event, 1),
                            self.distribution.left_censoring(time, Lambda),
                            self.distribution.right_censoring(time, Lambda))

        not_survival = 1 - survival
        # not_survival = tf.where(tf.equal(survival, 1), tf.ones(tf.shape(survival)) * 0.000001, 1 - survival)


        logloss = None
        if not sample_weights:
            logloss = tf.losses.log_loss(labels=event, predictions=not_survival)
        elif sample_weights == 'time':
            logloss = tf.losses.log_loss(labels=event, predictions=not_survival, weights=time)
        running_loss, loss_update = tf.metrics.mean(logloss)
        loss_mean = tf.reduce_mean(logloss)
        training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_mean)


        not_survival_binary = tf.where(tf.greater_equal(not_survival, 0.5),
                                       tf.ones(tf.shape(not_survival)),
                                       tf.zeros(tf.shape(not_survival)))


        running_auc, auc_update = None, None
        running_acc, auc_update = None, None
        if not sample_weights:
            running_auc, auc_update = tf.metrics.auc(labels=event, predictions=not_survival)
            running_acc, acc_update = tf.metrics.accuracy(labels=event, predictions=not_survival_binary)
        elif sample_weights == 'time':
            running_auc, auc_update = tf.metrics.auc(labels=event, predictions=not_survival, weights=time)
            running_acc, acc_update = tf.metrics.accuracy(labels=event, predictions=not_survival_binary, weights=time)


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
                    _, loss_batch, _, _, Lambda_batch, not_survival_batch = sess.run([training_op, loss_mean,
                                                                                      auc_update, acc_update, Lambda,
                                                                                      not_survival],
                                                                   feed_dict={input_vectors: features_batch,
                                                                              time: duration_batch,
                                                                              event: event_batch})

                    # print(Lambda_batch)
                    # print(survival_batch)
                    print("Epoch %d - Batch %d/%d: batch loss = %.4f" %
                          (epoch, num_batch, num_total_batches, loss_batch))



                # evaluation on training data
                eval_nodes_update = [loss_update, auc_update, acc_update, not_survival]
                eval_nodes_metric = [running_loss, running_auc, running_acc]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, auc_train, acc_train = self.evaluate(train_data.make_batch(self.batch_size),
                                                                 sess, eval_nodes_update, eval_nodes_metric)
                print("*** On Training Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_train, auc_train, acc_train))

                # evaluation on validation data
                loss_val, auc_val, acc_val = self.evaluate(val_data.make_batch(self.batch_size),
                                                           sess, eval_nodes_update, eval_nodes_metric)
                print("*** On Validation Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_val, auc_val, acc_val))

                # evaluation on test data
                loss_test, auc_test, acc_test = self.evaluate(test_data.make_batch(self.batch_size),
                                                              sess, eval_nodes_update, eval_nodes_metric)
                print("*** On Test Set:\tloss = %.6f\tauc = %.4f\taccuracy = %.4f" % (loss_test, auc_test, acc_test))



    def evaluate(self, next_batch, sess, updates, metrics):
        all_not_survival = []
        all_events = []
        for duration_batch, event_batch, features_batch in next_batch:
            _,_,_,not_survival  = sess.run(updates, feed_dict={'input_vectors:0': features_batch,
                                             'time:0': duration_batch,
                                             'event:0': event_batch})
            all_not_survival.extend(not_survival)
            all_events.extend(event_batch)
        all_not_survival = np.array(all_not_survival, dtype=np.float64)
        all_not_survival_bin = np.where(all_not_survival>=0.5, 1.0, 0.0)
        all_events = np.array(all_events, dtype=np.float64)
        print("SKLEARN:\tLOGLOSS = %.6f,\tAUC=%.4f,\tAccuracy=%.4f" % (log_loss(all_events, all_not_survival),
                                                                   roc_auc_score(all_events, all_not_survival),
                                                                   accuracy_score(all_events, all_not_survival_bin)))

        return sess.run(metrics)



if __name__ == "__main__":
    with open('../Vectors_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(distribution = WeibullDistribution(),
                    batch_size = 1000,
                    num_epochs = 20,
                    k = 1,
                    learning_rate = 0.01 )
    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('../Vectors_train.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_val.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_test.p', 'rb'))),
                    sample_weights=None)
