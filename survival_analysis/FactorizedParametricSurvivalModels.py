import numpy as np, pickle, csv

import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.DataReader import SurvivalData
from survival_analysis.Distributions import WeibullDistribution, LogLogisticDistribution

class FactorizedParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k, learning_rate=0.001,
                 lambda1=0.0, lambda2=0.0):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def linear_function(self, weights_linear):
        intercept = tf.Variable(0.1)
        return tf.reduce_sum(weights_linear, axis=-1) + intercept

    def factorization_machines(self, weights_factorized):
        dot_product_res = tf.matmul(weights_factorized, tf.transpose(weights_factorized, perm=[0,2,1]))
        element_product_res = weights_factorized * weights_factorized
        pairs_mulsum = tf.reduce_sum(tf.multiply(0.5, tf.reduce_sum(dot_product_res, axis=2)
                                        - tf.reduce_sum(element_product_res, axis=2)),
                            axis=-1)
        return pairs_mulsum


    def run_graph(self, num_features, train_data, val_data, test_data, sample_weights=None):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 1
        :return:
        '''
        # INPUTs
        feature_indice = tf.placeholder(tf.int32, name='feature_indice')
        feature_values = tf.placeholder(tf.float32, name='feature_values')

        time = tf.placeholder(tf.float32, shape=[None], name='time')
        event = tf.placeholder(tf.int32, shape=[None], name='event')

        # shape: (batch_size, max_nonzero_len)
        embeddings_linear = tf.Variable(tf.truncated_normal(shape=(num_features,), mean=0.0, stddev=0.1))
            # tf.truncated_normal(shape=(num_features,), mean=0.0, stddev=0.01))
        # shape: (batch_size, max_nonzero_len, k)
        embeddings_factorized = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=0.1))


        filtered_embeddings_linear = tf.nn.embedding_lookup(embeddings_linear, feature_indice) * feature_values
        filtered_embeddings_factorized = tf.nn.embedding_lookup(embeddings_factorized, feature_indice) * \
                                  tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])

        linear_term = self.linear_function(filtered_embeddings_linear)
        factorized_term = self.factorization_machines(filtered_embeddings_factorized)
        scale = tf.nn.softplus(linear_term + factorized_term)

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

        # L2 regularized sum of squares loss function over the embeddings
        lambda_linear = tf.constant(self.lambda1)
        lambda_factorized = tf.constant(self.lambda2)
        l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_linear, tf.pow(embeddings_linear, 2)),
                tf.reduce_sum(tf.multiply(lambda_factorized, tf.pow(embeddings_factorized, 2)), axis=-1)
            )
        )

        loss_mean = tf.reduce_mean(logloss) + l2_norm
        training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_mean)


        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


        with tf.Session() as sess:
            init.run()

            max_loss_val = None

            num_total_batches = int(np.ceil(train_data.num_instances / self.batch_size))
            for epoch in range(1, self.num_epochs + 1):
                sess.run(running_vars_initializer)
                # model training
                num_batch = 0
                for time_batch, event_batch, featidx_batch, featval_batch, max_nz_len in train_data.make_sparse_batch(self.batch_size):
                    # print(time_batch)
                    num_batch += 1
                    _, loss_batch, _, scale_batch, linear_term_batch, factorized_term_batch, prob_batch = sess.run([training_op, loss_mean,
                                                                  acc_update, scale, linear_term, factorized_term, not_survival_proba],
                                                                   feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'time:0': time_batch,
                                             'event:0': event_batch})

                    # print()
                    # print('linear_term')
                    # print(linear_term_batch)
                    # print('factorized_term')
                    # print(factorized_term_batch)
                    # print('scale')
                    # print(scale_batch)
                    # print('not_survival_prob')
                    # print(prob_batch)

                    if epoch == 1:
                        print("Epoch %d - Batch %d/%d: batch loss = %.4f" %
                              (epoch, num_batch, num_total_batches, loss_batch))


                # evaluation on training data
                eval_nodes_update = [loss_update, acc_update, not_survival_proba]
                eval_nodes_metric = [running_loss, running_acc]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                print('*** On Training Set:')
                (loss_train, acc_train), _, _, _ = self.evaluate(train_data.make_sparse_batch(self.batch_size),
                                                                 running_vars_initializer, sess,
                                                                 eval_nodes_update, eval_nodes_metric,
                                                                 sample_weights)
                print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_train, acc_train))

                # evaluation on validation data
                print('*** On Validation Set:')
                (loss_val, acc_val), not_survival_val, events_val, times_val = self.evaluate(val_data.make_sparse_batch(self.batch_size),
                                                           running_vars_initializer, sess,
                                                           eval_nodes_update, eval_nodes_metric,
                                                           sample_weights)
                print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_val, acc_val))

                # evaluation on test data
                print('*** On Test Set:')
                (loss_test, acc_test), _, _, _ = self.evaluate(test_data.make_sparse_batch(self.batch_size),
                                                              running_vars_initializer, sess,
                                                              eval_nodes_update, eval_nodes_metric,
                                                              sample_weights)
                print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))

                if max_loss_val is None or loss_val < max_loss_val:
                    max_loss_val = loss_val
                    with open('../all_predictions_factorized.csv', 'w', newline="\n") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(('NOT_SURV_PROB', 'EVENTS', 'TIMES'))
                        for p, e, t in zip(not_survival_val, events_val, times_val):
                            csv_writer.writerow((p, e, t))
                    print('All predictions are outputted for error analysis')



    def evaluate(self, next_batch, running_init, sess, updates, metrics, sample_weights=None):
        all_not_survival = []
        all_events = []
        all_times = []
        sess.run(running_init)
        for time_batch, event_batch, featidx_batch, featval_batch, max_nz_len in next_batch:
            _, _, not_survival  = sess.run(updates, feed_dict={
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

        if not sample_weights:
            print("SKLEARN:\tLOGLOSS = %.6f,\tAccuracy = %.4f" % (log_loss(all_events, all_not_survival),
                                                                   accuracy_score(all_events, all_not_survival_bin)))
        elif sample_weights == 'time':
            print("SKLEARN:\tLOGLOSS = %.6f,\tAccuracy = %.4f" % (log_loss(all_events, all_not_survival,
                                                                                    sample_weight=all_times),
                                                                   accuracy_score(all_events, all_not_survival_bin,
                                                                                  sample_weight=all_times)))
        return sess.run(metrics), all_not_survival, all_events, all_times



if __name__ == "__main__":
    with open('../Vectors_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = FactorizedParametricSurvival(distribution = WeibullDistribution(),
                    batch_size = 128,
                    num_epochs = 20,
                    k = 30,
                    learning_rate=0.0001,
                    lambda1=0.0,
                    lambda2=0.0
                    )
    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('../Vectors_train.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_val.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../Vectors_test.p', 'rb'))),
                    sample_weights='time')
