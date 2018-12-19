import numpy as np, pickle, csv

import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.DataReader import SurvivalData
from survival_analysis import Distributions
from survival_analysis.EvaluationMetrics import c_index
from time import time as nowtime

class ParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k, learning_rate=0.001,
                 lambda_linear=0.0, lambda_factorized=0.0, lambda_hb_adxwon=0.0, lambda_hb_adxlose=0.0):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_linear = lambda_linear
        self.lambda_factorized = lambda_factorized
        self.lambda_hb_adxwon = lambda_hb_adxwon
        self.lambda_hb_adxlose = lambda_hb_adxlose

    def linear_function(self, weights_linear, intercept):
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
        :param k: the dimensionality of the embedding, Must be >= 0; when k=0, it is a simple model; Otherwise it is factorized
        :return:
        '''
        # INPUTs
        feature_indice = tf.placeholder(tf.int32, name='feature_indice')
        feature_values = tf.placeholder(tf.float32, name='feature_values')

        min_hbs = tf.placeholder(tf.float32, name='min_headerbids')  # for regularization
        max_hbs = tf.placeholder(tf.float32, name='max_headerbids')  # for regularization

        times = tf.placeholder(tf.float32, shape=[None], name='times')
        events = tf.placeholder(tf.int32, shape=[None], name='events')

        # shape: (batch_size, max_nonzero_len)
        embeddings_linear = tf.Variable(tf.truncated_normal(shape=(num_features,), mean=0.0, stddev=1e-5))
        filtered_embeddings_linear = tf.nn.embedding_lookup(embeddings_linear, feature_indice) * feature_values
        intercept = tf.Variable(1e-5)
        linear_term = self.linear_function(filtered_embeddings_linear, intercept)
        scale = linear_term

        embeddings_factorized = None
        if self.k > 0:
            # shape: (batch_size, max_nonzero_len, k)
            embeddings_factorized = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=1e-5))
            filtered_embeddings_factorized = tf.nn.embedding_lookup(embeddings_factorized, feature_indice) * \
                                      tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])
            factorized_term = self.factorization_machines(filtered_embeddings_factorized)
            scale += factorized_term


        scale = tf.nn.softplus(scale)

        # scale = tf.Variable(0.1)

        ''' 
        if event == 0, right-censoring
        if event == 1, left-censoring 
        '''
        not_survival_proba = self.distribution.left_censoring(times, scale)  # the left area


        not_survival_bin = tf.where(tf.greater_equal(not_survival_proba, 0.5),
                                    tf.ones(tf.shape(not_survival_proba)),
                                    tf.zeros(tf.shape(not_survival_proba)))

        running_acc, acc_update = None, None
        if not sample_weights:
            running_acc, acc_update = tf.metrics.accuracy(labels=events, predictions=not_survival_bin)
        elif sample_weights == 'time':
            running_acc, acc_update = tf.metrics.accuracy(labels=events, predictions=not_survival_bin, weights=times)

        batch_loss = None
        if not sample_weights:
            batch_loss = tf.losses.log_loss(labels=events, predictions=not_survival_proba,
                                            reduction = tf.losses.Reduction.MEAN)
        elif sample_weights == 'time':
            batch_loss = tf.losses.log_loss(labels=events, predictions=not_survival_proba, weights=times,
                                            reduction = tf.losses.Reduction.MEAN)
        running_loss, loss_update = tf.metrics.mean(batch_loss)


        # Header Bidding Regularization
        hb_adxwon_partitions = tf.cast(
            tf.logical_and(tf.equal(events, 0),  # adx won
                           tf.logical_and(
                               tf.not_equal(0.0, max_hbs),  # the max_hb is not missing
                               tf.less(times, max_hbs)
                               # tf.less(times, min_hbs),
                               # tf.logical_and(
                               #                                #     # tf.less(times, max_hbs),  # the max hb > the revenue
                               #                                #                # tf.less(max_hbs - time, 1.0)  # remove the outliers
                               #                                #                tf.less(times, min_hbs),
                               #                                #                tf.less((max_hbs - times) / times, 0.01)
                               #                                #                # tf.logical_and(
                               #                                #                #     tf.less((max_hbs - times) / times, 0.01),
                               #                                #                #     tf.less(times, 10.0)
                               #                                #                # )
                               #                                #                )
                           )
                           ), tf.int32)
        hb_adxlose_partitions = tf.cast(
            tf.logical_and(tf.equal(events, 1),  # adx lose
                           tf.logical_and(
                               tf.not_equal(0.0, min_hbs),  # the min_hb is not missing
                               tf.less(min_hbs, times)  # the min hb < the floor
                               # tf.less(max_hbs, times),
                               # tf.logical_and(
                               #                tf.less(min_hbs, times),
                               #                # tf.less(max_hbs - time, 1.0)  # remove the outliers
                               #                tf.less(0.9, (times - min_hbs) / times)
                               #                # tf.logical_and(
                               #                #     tf.less(0.1, (times - min_hbs) / times),
                               #                #     tf.less(times, 10.0)
                               #                # )
                               #                )
                           )
                           ), tf.int32)

        # Using boolean_mask instead of dynamic_partition leads to:
        # "UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory."
        # https://stackoverflow.com/questions/44380727/get-userwarning-while-i-use-tf-boolean-mask?noredirect=1&lq=1
        regable_hb_adxwon = tf.dynamic_partition(max_hbs, hb_adxwon_partitions, 2)[1]
        regable_hb_adxlose = tf.dynamic_partition(min_hbs, hb_adxlose_partitions, 2)[1]
        regable_scale_adxwon = tf.dynamic_partition(scale, hb_adxwon_partitions, 2)[1]
        regable_scale_adxlose = tf.dynamic_partition(scale, hb_adxlose_partitions, 2)[1]

        hb_adxwon_pred = self.distribution.left_censoring(regable_hb_adxwon, regable_scale_adxwon)
        hb_adxlose_pred = self.distribution.left_censoring(regable_hb_adxlose, regable_scale_adxlose)

        hb_reg_adxwon, hb_reg_adxlose = None, None
        if not sample_weights:
        # if True:
            hb_reg_adxwon = tf.losses.log_loss(labels=tf.zeros(tf.shape(hb_adxwon_pred)),
                                               predictions=hb_adxwon_pred)
            hb_reg_adxlose = tf.losses.log_loss(labels=tf.zeros(tf.shape(hb_adxlose_pred)),
                                                predictions=hb_adxlose_pred)
        elif sample_weights == 'time':
            regable_time_adxwon = tf.dynamic_partition(times, hb_adxwon_partitions, 2)[1]
            regable_time_adxlose = tf.dynamic_partition(times, hb_adxlose_partitions, 2)[1]
            hb_reg_adxwon = tf.losses.log_loss(labels=tf.ones(tf.shape(hb_adxwon_pred)),
                                               predictions=hb_adxwon_pred,
                                               weights=1.0 / regable_time_adxwon)
            hb_reg_adxlose = tf.losses.log_loss(labels=tf.zeros(tf.shape(hb_adxlose_pred)),
                                                predictions=hb_adxlose_pred,
                                                weights=1.0 / regable_time_adxlose)
        mean_hb_reg_adxwon = tf.reduce_mean(hb_reg_adxwon)
        mean_hb_reg_adxlose = tf.reduce_mean(hb_reg_adxlose)


        # L2 regularized sum of squares loss function over the embeddings
        l2_norm = tf.constant(self.lambda_linear) * tf.pow(embeddings_linear, 2)
        if embeddings_factorized is not None:
            l2_norm += tf.reduce_sum(tf.pow(embeddings_factorized, 2), axis=-1)
        sum_l2_norm = tf.constant(self.lambda_factorized) * tf.reduce_sum(l2_norm)


        loss_mean = batch_loss + \
                    tf.constant(self.lambda_hb_adxwon) * mean_hb_reg_adxwon + \
                    tf.constant(self.lambda_hb_adxlose) * mean_hb_reg_adxlose + \
                    sum_l2_norm
        # training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_mean)

        ### gradient clipping
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss_mean))
        gradients_clipped, _ = tf.clip_by_global_norm(gradients, 5.0)
        training_op = optimizer.apply_gradients(zip(gradients_clipped, variables))


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
                start = nowtime()
                for time_batch, event_batch, featidx_batch, featval_batch, minhbs_natch, maxhbs_batch, max_nz_len \
                        in train_data.make_sparse_batch(self.batch_size):

                    num_batch += 1

                    _, loss_batch, _, event_batch, time_batch = sess.run([training_op, loss_mean,
                                                                  acc_update, events, times],
                                                                   feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'min_headerbids:0': minhbs_natch,
                                             'max_headerbids:0': maxhbs_batch,
                                             'times:0': time_batch,
                                             'events:0': event_batch})

                    # print()
                    # print('mean_hb_reg_adxwon_batch')
                    # print(mean_hb_reg_adxwon_batch)
                    # print('mean_hb_reg_adxlose_batch')
                    # print(mean_hb_reg_adxlose_batch)
                    # print('mean_batch_loss_batch')
                    # print(mean_batch_loss_batch)
                    # print("event_batch")
                    # print(event_batch)
                    # print('time_batch')
                    # print(time_batch)

                    if epoch == 1:
                        print("Epoch %d - Batch %d/%d: batch loss = %.4f" %
                              (epoch, num_batch, num_total_batches, loss_batch))
                        print("                         time: %.4fs" % (nowtime() - start))
                        start = nowtime()


                # evaluation on training data
                eval_nodes_update = [loss_update, acc_update, not_survival_proba]
                eval_nodes_metric = [running_loss, running_acc]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                print('*** On Training Set:')
                (loss_train, acc_train), _, _, _ = self.evaluate(train_data.make_sparse_batch(),
                                                                 running_vars_initializer, sess,
                                                                 eval_nodes_update, eval_nodes_metric,
                                                                 sample_weights)
                # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_train, acc_train))

                # evaluation on validation data
                print('*** On Validation Set:')
                (loss_val, acc_val), not_survival_val, events_val, times_val = self.evaluate(val_data.make_sparse_batch(),
                                                           running_vars_initializer, sess,
                                                           eval_nodes_update, eval_nodes_metric,
                                                           sample_weights)
                # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_val, acc_val))
                print("Validation C-Index = %.4f" % c_index(events_val, not_survival_val, times_val))



                if max_loss_val is None or loss_val < max_loss_val:
                    print("!!! GET THE LOWEST VAL LOSS !!!")
                    max_loss_val = loss_val

                    # evaluation on test data
                    print('*** On Test Set:')
                    (loss_test, acc_test), not_survival_test, events_test, times_test = self.evaluate(
                        test_data.make_sparse_batch(),
                        running_vars_initializer, sess,
                        eval_nodes_update, eval_nodes_metric,
                        sample_weights)
                    # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))
                    print("TEST C-Index = %.4f" % c_index(events_test, not_survival_test, times_test))


                    # Store prediction results
                    with open('../all_predictions_factorized.csv', 'w', newline="\n") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(('NOT_SURV_PROB', 'EVENTS', 'TIMES'))
                        for p, e, t in zip(not_survival_val, events_val, times_val):
                            csv_writer.writerow((p, e, t))
                    print('All predictions are outputted for error analysis')

                    # Store parameters
                    params = {'embeddings_linear': embeddings_linear.eval(),
                              'intercept': intercept.eval(),
                              'shape': self.distribution.shape,
                              'distribution_name': type(self.distribution).__name__}
                    if embeddings_factorized is not None:
                        params['embeddings_factorized'] = embeddings_factorized.eval(),
                    pickle.dump(params, open('../params_k%d.pkl' % self.k, 'wb'))





    def evaluate(self, next_batch, running_init, sess, updates, metrics, sample_weights=None):
        all_not_survival = []
        all_events = []
        all_times = []
        sess.run(running_init)
        for time_batch, event_batch, featidx_batch, featval_batch, minhbs_natch, maxhbs_batch, max_nz_len in next_batch:
            _, _, not_survival  = sess.run(updates, feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'min_headerbids:0': minhbs_natch,
                                             'max_headerbids:0': maxhbs_batch,
                                             'times:0': time_batch,
                                             'events:0': event_batch})
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
    with open('../FeatVec_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(
        distribution=Distributions.WeibullDistribution(),
        batch_size=2048,
        num_epochs=10,
        k=80,
        learning_rate=1e-3,
        lambda_linear=1e-7,
        lambda_factorized=1e-7,
        lambda_hb_adxwon=1e-4,
        lambda_hb_adxlose=1e-4
    )

    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('../TRAIN_SET.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../VAL_SET.p', 'rb'))),
                    SurvivalData(*pickle.load(open('../TEST_SET.p', 'rb'))),
                    sample_weights='time')
