import numpy as np, pickle

import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.DataReader import SurvivalData
from survival_analysis.Distributions import WeibullDistribution, LogLogisticDistribution

class ParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k, learning_rate=0.01):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate

    def factorization_machines(self, weights_one, weights_two):
        dot_product_res = tf.matmul(weights_two, tf.transpose(weights_two))
        element_product_res = weights_two * weights_two
        pairs_mulsum = tf.reduce_sum(tf.multiply(0.5, tf.reduce_sum(dot_product_res, axis=2)
                                        - tf.reduce_sum(element_product_res, axis=2)),
                            axis=-1)
        return pairs_mulsum + tf.reduce_sum(weights_one, axis=-1)


    def run_graph(self, num_features, train_data, val_data, test_data, sample_weights=None):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 1
        :return:
        '''
        # INPUTs
        max_nonzero_len = tf.placeholder(tf.int32,
                                  shape=[],
                                  name='max_nonzero_len')
        feature_indice = tf.placeholder(tf.float32,
                                  shape=[None, max_nonzero_len],
                                  name='feature_indice')
        feature_values = tf.placeholder(tf.float32,
                                  shape=[None, max_nonzero_len],
                                  name='feature_values')

        time = tf.placeholder(tf.float32, shape=[None], name='time')
        event = tf.placeholder(tf.int32, shape=[None], name='event')

        # shape: (batch_size, max_nonzero_len)
        embeddings_one = tf.Variable(tf.truncated_normal(shape=(num_features), mean=0.0, stddev=0.02))
        # shape: (batch_size, max_nonzero_len, k)
        embeddings_two = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=0.02))

        w0 = tf.Variable(0.0)

        filtered_embeddings_one = tf.nn.embedding_lookup(embeddings_one, feature_indice) * feature_values + w0
        filtered_embeddings_two = tf.nn.embedding_lookup(embeddings_two, feature_indice) * \
                                  tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])


        scale = tf.exp(self.factorization_machines(filtered_embeddings_one, filtered_embeddings_two))

        ''' 
        if event == 0, right-censoring
        if event == 1, left-censoring 
        '''
        not_survival_proba = self.distribution.left_censoring(time, scale)  # the left area
        # survival_proba = self.distribution.right_censoring(time, scale)  # the right area

        # event_pred = tf.stack([survival_proba, not_survival_proba], axis=1)

        # predictions = tf.where(tf.equal(event, 1),
        #                     self.distribution.left_censoring(time, scale),
        #                     self.distribution.right_censoring(time, scale))
        # neg_log_likelihood = -1 * tf.reduce_sum(tf.log(predictions))

        not_survival_bin = tf.where(tf.greater_equal(not_survival_proba, 0.5),
                                    tf.ones(tf.shape(not_survival_proba)),
                                    tf.zeros(tf.shape(not_survival_proba)))

        running_acc, acc_update = None, None
        if not sample_weights:
            running_acc, acc_update = tf.metrics.accuracy(labels=event, predictions=not_survival_bin)
        elif sample_weights == 'time':
            running_acc, acc_update = tf.metrics.accuracy(labels=event, predictions=not_survival_bin, weights=time)

        logloss = None
        if not sample_weights:
            logloss = tf.losses.log_loss(labels=event, predictions=not_survival_proba)
        elif sample_weights == 'time':
            logloss = tf.losses.log_loss(labels=event, predictions=not_survival_proba, weights=time)
        running_loss, loss_update = tf.metrics.mean(logloss)
        loss_mean = tf.reduce_mean(logloss)
        training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_mean)


        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


        with tf.Session() as sess:
            init.run()

            num_total_batches = int(np.ceil(train_data.num_instances / self.batch_size))
            for epoch in range(1, self.num_epochs + 1):
                sess.run(running_vars_initializer)
                # model training
                num_batch = 0
                for time_batch, event_batch, featidx_batch, featval_batch, max_nz_len in train_data.make_sparse_batch(self.batch_size):
                    # print(time_batch)
                    num_batch += 1
                    _, loss_batch, _, scale_batch = sess.run([training_op, loss_mean,
                                                                  acc_update, scale,
                                                                                    ],
                                                                   feed_dict={
                                            'max_nonzero_len:0': max_nz_len,
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'time:0': time_batch,
                                             'event:0': event_batch})


                    if epoch == 1:
                        print("Epoch %d - Batch %d/%d: batch loss = %.4f" %
                              (epoch, num_batch, num_total_batches, loss_batch))


                # evaluation on training data
                eval_nodes_update = [loss_update, acc_update, not_survival_proba]
                eval_nodes_metric = [running_loss, running_acc]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, acc_train = self.evaluate(train_data.make_sparse_batch(self.batch_size),
                                                                 running_vars_initializer, sess,
                                                                 eval_nodes_update, eval_nodes_metric,
                                                                 sample_weights)
                print("*** On Training Set:\tloss = %.6f\taccuracy = %.4f" % (loss_train, acc_train))

                # evaluation on validation data
                loss_val, acc_val = self.evaluate(val_data.make_sparse_batch(self.batch_size),
                                                           running_vars_initializer, sess,
                                                           eval_nodes_update, eval_nodes_metric,
                                                           sample_weights)
                print("*** On Validation Set:\tloss = %.6f\taccuracy = %.4f" % (loss_val, acc_val))

                # evaluation on test data
                loss_test, acc_test = self.evaluate(test_data.make_sparse_batch(self.batch_size),
                                                              running_vars_initializer, sess,
                                                              eval_nodes_update, eval_nodes_metric,
                                                              sample_weights)
                print("*** On Test Set:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))



    def evaluate(self, next_batch, running_init, sess, updates, metrics, sample_weights=None):
        all_not_survival = []
        all_events = []
        all_times = []
        sess.run(running_init)
        for time_batch, event_batch, featidx_batch, featval_batch, max_nz_len in next_batch:
            _, _, not_survival  = sess.run(updates, feed_dict={
                                             'max_nonzero_len:0': max_nz_len,
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'time:0': time_batch,
                                             'event:0': event_batch})
            all_not_survival.extend(not_survival)
            all_events.extend(event_batch)
            all_times.extend(time_batch)

        all_not_survival = np.array(all_not_survival, dtype=np.float64)
        all_not_survival_bin = np.where(all_not_survival>=0.5, 1.0, 0.0)
        all_events = np.array(all_events, dtype=np.float64)
        # print(all_not_survival_bin)
        # print(all_events)
        # print(sum(1 for i in range(len(all_events)) if all_not_survival_bin[i] == all_events[i]))
        # print(len(all_events))
        if not sample_weights:
            print("SKLEARN:\tLOGLOSS = %.6f,\tAUC = %.4f,\tAccuracy = %.4f" % (log_loss(all_events, all_not_survival),
                                                                   roc_auc_score(all_events, all_not_survival),
                                                                   accuracy_score(all_events, all_not_survival_bin)))
        elif sample_weights == 'time':
            print("SKLEARN:\tLOGLOSS = %.6f,\tAUC = %.4f,\tAccuracy = %.4f" % (log_loss(all_events, all_not_survival,
                                                                                    sample_weight=all_times),
                                                                   roc_auc_score(all_events, all_not_survival,
                                                                                 sample_weight=all_times),
                                                                   accuracy_score(all_events, all_not_survival_bin,
                                                                                  sample_weight=all_times)))
        return sess.run(metrics)



if __name__ == "__main__":
    with open('../Vectors_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(distribution = LogLogisticDistribution(),
                    batch_size = 512,
                    num_epochs = 20,
                    k = 10,
                    learning_rate = 0.005 )
    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('../Vectors_train.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_val.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_test.p', 'rb'))),
                    sample_weights='time')
