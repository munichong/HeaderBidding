import sys
from time import time as nowtime
import tensorflow as tf
import numpy as np, pickle, csv
from sklearn.metrics import log_loss, accuracy_score

from failure_rate_prediction_conf.DataReader import SurvivalData
from failure_rate_prediction_conf import Distributions
from failure_rate_prediction_conf.EvaluationMetrics import c_index


ONLY_FREQ_TRAIN = False
ONLY_FREQ_TEST = False

ONLY_HB_IMP = False

MIN_OCCURRENCE = 5

class ParametricSurvival:

    def __init__(self, distribution, batch_size, num_epochs, k, learning_rate=0.001,
                 lambda_linear=0.0, lambda_factorized=0.0,
                 importance_failure_rate_optimize=0.0, importance_optimal_reserve_optimize=0.0):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_linear = lambda_linear
        self.lambda_factorized = lambda_factorized
        self.importance_failure_rate_optimize = importance_failure_rate_optimize
        self.importance_optimal_reserve_optimize = importance_optimal_reserve_optimize

    def linear_function(self, weights_linear, intercept):
        return tf.reduce_sum(weights_linear, axis=-1) + intercept

    def factorization_machines(self, weights_factorized):
        dot_product_res = tf.matmul(weights_factorized, tf.transpose(weights_factorized, perm=[0,2,1]))
        element_product_res = weights_factorized * weights_factorized
        pairs_mulsum = tf.reduce_sum(tf.multiply(0.5, tf.reduce_sum(dot_product_res, axis=2)
                                        - tf.reduce_sum(element_product_res, axis=2)),
                            axis=-1)
        return pairs_mulsum

    def initialize_fm_weights(self, trainable=True):
        # shape: (batch_size, max_nonzero_len)
        embeddings_linear = tf.get_variable('embeddings_linear',
                                            initializer=tf.truncated_normal(shape=(num_features,), mean=0.0, stddev=1e-5),
                                            trainable=trainable)
        embeddings_factorized = None
        if self.k > 0:
            # shape: (batch_size, max_nonzero_len, k)
            embeddings_factorized = tf.get_variable('embeddings_factorized',
                                                    initializer=tf.truncated_normal(shape=(num_features, self.k),
                                                                                    mean=0.0, stddev=1e-5),
                                                    trainable=trainable)
        intercept = tf.get_variable('fm_intercept', initializer=1e-5, trainable=trainable)
        return embeddings_linear, embeddings_factorized, intercept

    def initialize_optimal_reserve(self, trainable=True):
        return tf.get_variable('opt_res',
                               initializer=tf.truncated_normal(shape=[self.batch_size], mean=0.0, stddev=1e-1),
                               validate_shape=False,
                               trainable=trainable)

    def calculate_scale(self, embeddings_linear, embeddings_factorized, feature_indice, feature_values, intercept):
        filtered_embeddings_linear = tf.nn.embedding_lookup(embeddings_linear, feature_indice) * feature_values
        scales = self.linear_function(filtered_embeddings_linear, intercept)

        filtered_embeddings_factorized = None
        if self.k > 0:
            filtered_embeddings_factorized = tf.nn.embedding_lookup(embeddings_factorized, feature_indice) * \
                                             tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])
            factorized_term = self.factorization_machines(filtered_embeddings_factorized)
            scales += factorized_term

        scales = tf.nn.softplus(scales)
        return scales, filtered_embeddings_linear, filtered_embeddings_factorized

    def compute_failure_rate(self, hist_res, scales, shape):
        '''
        if event == 0, right-censoring
        if event == 1, left-censoring
        '''
        not_survival_proba = self.distribution.left_censoring(hist_res, scales, shape)  # the left area
        not_survival_bin = tf.where(tf.greater_equal(not_survival_proba, 0.5),
                                    tf.ones(tf.shape(not_survival_proba)),
                                    tf.zeros(tf.shape(not_survival_proba)))
        return not_survival_proba, not_survival_bin

    def compute_lower_bound_expected_revenue(self, opt_reserves, scales, max_hbs, shape):
        optimal_failure_proba, _ = self.compute_failure_rate(opt_reserves, scales, shape)
        return (1 - optimal_failure_proba) * opt_reserves + optimal_failure_proba * max_hbs

    def wrong_side_large_penalty(self, larger_reserves, less_reserves):
        return tf.square(larger_reserves - less_reserves)

    def right_side_small_penalty(self, larger_reserves, less_reserves):
        return tf.zeros(tf.shape(larger_reserves))

    def compute_optimal_reserve_error(self, opt_reserves, hist_reserves, was_failed):
        loss1 = tf.where(tf.logical_and(tf.equal(was_failed, 1),
                                        tf.greater_equal(opt_reserves, hist_reserves)),
                        self.wrong_side_large_penalty(opt_reserves, hist_reserves),
                        tf.zeros(tf.shape(was_failed)))
        loss2 = tf.where(tf.logical_and(tf.equal(was_failed, 1),
                                        tf.less(opt_reserves, hist_reserves)),
                        self.right_side_small_penalty(hist_reserves, opt_reserves),
                        tf.zeros(tf.shape(was_failed)))
        loss3 = tf.where(tf.logical_and(tf.equal(was_failed, 0),
                                        tf.less_equal(opt_reserves, hist_reserves)),
                        self.wrong_side_large_penalty(hist_reserves, opt_reserves),
                        tf.zeros(tf.shape(was_failed)))
        loss4 = tf.where(tf.logical_and(tf.equal(was_failed, 0),
                                        tf.greater(opt_reserves, hist_reserves)),
                         self.right_side_small_penalty(opt_reserves, hist_reserves),
                         tf.zeros(tf.shape(was_failed)))
        return tf.reduce_mean(loss1 + loss2 + loss3 + loss4)


    def run_graph(self, num_features, train_data, val_data, test_data, sample_weights=None):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 0; when k=0, it is a simple model; Otherwise it is factorized
        :return:
        '''
        ''' ================= INPUTs ================== '''
        feature_indice = tf.placeholder(tf.int32, name='feature_indice')
        feature_values = tf.placeholder(tf.float32, name='feature_values')

        min_hbs = tf.placeholder(tf.float32, name='min_headerbids')  # for regularization
        max_hbs = tf.placeholder(tf.float32, name='max_headerbids')  # for regularization

        hist_reserve_prices = tf.placeholder(tf.float32, shape=[None], name='times')  # historical reserve price
        was_failed = tf.placeholder(tf.int32, shape=[None], name='events')  # if the historical reserve price was failed to be outbid

        ''' ================= Initialize FM Weights ================== '''
        embeddings_linear, embeddings_factorized, fm_intercept = self.initialize_fm_weights()

        ''' ================= Calculate Scale ================== '''
        scales, filtered_embeddings_linear, filtered_embeddings_factorized = \
            self.calculate_scale(embeddings_linear, embeddings_factorized, feature_indice, feature_values, fm_intercept)

        ''' ================= Calculate Historical Failure Rate ================== '''
        shape = tf.get_variable('dist_shape', initializer=0.2, trainable=True)
        hist_failure_proba, hist_failure_bin = self.compute_failure_rate(hist_reserve_prices, scales, shape)

        ''' =========== Calculate The Loss of Historical Failure Rate Prediction =========== '''
        failure_rate_running_acc, failure_rate_acc_update = None, None
        if not sample_weights:
            failure_rate_running_acc, failure_rate_acc_update = tf.metrics.accuracy(labels=was_failed, predictions=hist_failure_bin)
        elif sample_weights == 'time':
            failure_rate_running_acc, failure_rate_acc_update = tf.metrics.accuracy(labels=was_failed, predictions=hist_failure_bin,
                                                          weights=hist_reserve_prices)

        failure_rate_loss = None
        if not sample_weights:
            failure_rate_loss = tf.losses.log_loss(labels=was_failed, predictions=hist_failure_proba,
                                            reduction=tf.losses.Reduction.MEAN)
        elif sample_weights == 'time':
            failure_rate_loss = tf.losses.log_loss(labels=was_failed, predictions=hist_failure_proba, weights=hist_reserve_prices,
                                            reduction=tf.losses.Reduction.MEAN)
        failure_rate_running_loss, failure_rate_loss_update = tf.metrics.mean(failure_rate_loss)


        ''' ================= Calculate Lower Bound Expected Revenue ================== '''
        optimal_reserve_prices_raw = self.initialize_optimal_reserve()
        optimal_reserve_prices_positive = tf.square(optimal_reserve_prices_raw)
        lower_bound_expected_revenue = self.compute_lower_bound_expected_revenue(optimal_reserve_prices_positive, scales, max_hbs, shape)
        lower_bound_expected_revenue = tf.where(tf.is_nan(lower_bound_expected_revenue),
                                                tf.zeros(tf.shape(lower_bound_expected_revenue)),
                                                lower_bound_expected_revenue)
        expected_revenue_mean_loss = -1 * tf.reduce_mean(lower_bound_expected_revenue)


        ''' =============== Optimize Lower Bound Expected Revenue ============== '''
        expected_revenue_optimizer = tf.train.AdamOptimizer(
            learning_rate=1
        ).minimize(
            loss=expected_revenue_mean_loss,
            var_list=[optimal_reserve_prices_raw]
        )


        ''' =============== Calculate the Loss of Wrong-Side Optimal Revenue =============== '''
        optimal_reserve_loss = self.compute_optimal_reserve_error(optimal_reserve_prices_positive,
                                                                  hist_reserve_prices,
                                                                  was_failed)

        ''' ============= L2 regularized sum of squares loss function over the embeddings ============= '''
        l2_norm = self.lambda_linear * tf.nn.l2_loss(filtered_embeddings_linear)
        if embeddings_factorized is not None:
            l2_norm += self.lambda_factorized * tf.nn.l2_loss(filtered_embeddings_factorized)

        ''' ====================== Combine and Optimize All Losses ======================= '''
        combined_mean_loss = expected_revenue_mean_loss + \
                     self.importance_failure_rate_optimize * failure_rate_loss + \
                     self.importance_optimal_reserve_optimize * optimal_reserve_loss

        combined_training_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        ).minimize(
            loss=combined_mean_loss,
            var_list=[embeddings_linear,
                      embeddings_factorized,
                      fm_intercept,
                      shape]
        )

        ### gradient clipping
        # combined_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # gradients, variables = zip(*combined_optimizer.compute_gradients(combined_mean_loss,
        #                                                                  var_list=[embeddings_linear,
        #                                                                            embeddings_factorized,
        #                                                                            fm_intercept, shape]))
        # gradients_clipped, _ = tf.clip_by_global_norm(gradients, 100)
        # combined_training_optimizer = combined_optimizer.apply_gradients(zip(gradients_clipped, variables))


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
                for time_batch, event_batch, featidx_batch, featval_batch, minhbs_batch, maxhbs_batch, max_nz_len \
                        in train_data.make_sparse_batch(self.batch_size, only_freq=ONLY_FREQ_TRAIN):

                    if time_batch.shape[0] < self.batch_size:
                        print("Skip the batch whose size is %d < %d" % (time_batch.shape[0], self.batch_size))
                        break

                    num_batch += 1

                    prev_batch_expected_revenue_mean_loss = None
                    while True:
                        _, batch_expected_revenue_mean_loss, batch_optimal_reserve_prices = \
                            sess.run([expected_revenue_optimizer,
                                      expected_revenue_mean_loss,
                                      optimal_reserve_prices_positive
                                      ],
                                     feed_dict={
                                         'feature_indice:0': featidx_batch,
                                         'feature_values:0': featval_batch,
                                         'min_headerbids:0': minhbs_batch,
                                         'max_headerbids:0': maxhbs_batch,
                                         'times:0': time_batch,
                                         'events:0': event_batch})

                        print('\tbatch_expected_revenue_mean_loss = %.4f, optimal_reserve_prices = %s' %
                              (batch_expected_revenue_mean_loss, batch_optimal_reserve_prices))
                        if (prev_batch_expected_revenue_mean_loss != None and
                                (batch_expected_revenue_mean_loss - prev_batch_expected_revenue_mean_loss) / prev_batch_expected_revenue_mean_loss < 0.01):
                            break
                        prev_batch_expected_revenue_mean_loss = batch_expected_revenue_mean_loss



                    _, combined_loss_batch, batch_hist_failure_proba, batch_lower_bound_expected_revenue, batch_failure_rate_loss, batch_opt_reserve_loss, batch_scales, batch_embed_linear, batch_embed_factorized = sess.run([
                        combined_training_optimizer,
                        combined_mean_loss,
                        hist_failure_proba,
                        lower_bound_expected_revenue,
                        failure_rate_loss,
                        optimal_reserve_loss,
                        scales,
                        filtered_embeddings_linear,
                        filtered_embeddings_factorized
                    ],
                                                      feed_dict={
                                                          'feature_indice:0': featidx_batch,
                                                          'feature_values:0': featval_batch,
                                                          'min_headerbids:0': minhbs_batch,
                                                          'max_headerbids:0': maxhbs_batch,
                                                          'times:0': time_batch,
                                                          'events:0': event_batch})

                    print()
                    print('batch_embed_linear', np.isnan(batch_embed_linear).any())
                    print('batch_embed_factorized', np.isnan(batch_embed_factorized).any())
                    print('scales:', np.isnan(batch_scales).any())
                    print('hist_failure_proba:', np.isnan(batch_hist_failure_proba).any())
                    print('failure_rate_loss:', np.isnan(batch_failure_rate_loss).any())
                    print('optimal_reserve_loss:', np.isnan(batch_opt_reserve_loss).any())

                    if epoch == 1:
                        print("Epoch %d - Batch %d/%d: combined batch loss = %.4f" %
                              (epoch, num_batch, num_total_batches, combined_loss_batch))
                        print("                         expected revenue = %.4f" % batch_lower_bound_expected_revenue)
                        print("                         time: %.4fs" % (nowtime() - start))
                        start = nowtime()

                    print('*********************************\n\n')

                # evaluation on training data
                eval_nodes_update = [failure_rate_loss_update, failure_rate_acc_update, hist_failure_proba, scales, max_hbs]
                eval_nodes_metric = [failure_rate_running_loss, failure_rate_running_acc]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                print('*** On Training Set:')
                (loss_train, acc_train), _, _, _, _, _ = self.evaluate(train_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST),
                                                                 running_vars_initializer, sess,
                                                                 eval_nodes_update, eval_nodes_metric,
                                                                 sample_weights)
                # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_train, acc_train))

                # evaluation on validation data
                print('*** On Validation Set:')
                (loss_val, acc_val), not_survival_val, _, _, events_val, times_val = self.evaluate(val_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST),
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
                    (loss_test, acc_test), not_survival_test, scale_test, max_hbs_test, events_test, times_test = self.evaluate(
                        test_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST),
                        running_vars_initializer, sess,
                        eval_nodes_update, eval_nodes_metric,
                        sample_weights)
                    # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))
                    print("TEST C-Index = %.4f" % c_index(events_test, not_survival_test, times_test))


                    # Store prediction results
                    with open('output/all_predictions_factorized.csv', 'w', newline="\n") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(('NOT_SURV_PROB', 'EVENTS', 'MAX(RESERVE, REVENUE)', 'MAX_HB', 'SCALE'))
                        for p, e, t, h, sc in zip(not_survival_test, events_test, times_test, max_hbs_test, scale_test):
                            csv_writer.writerow((p, e, t, h, sc))
                    print('All predictions are outputted for error analysis')

                    # # Store parameters
                    # params = {'embeddings_linear': embeddings_linear.eval(),
                    #           'intercept': intercept.eval(),
                    #           'shape': shape.eval(),
                    #           'distribution_name': type(self.distribution).__name__}
                    # if embeddings_factorized is not None:
                    #     params['embeddings_factorized'] = embeddings_factorized.eval(),
                    # pickle.dump(params, open('output/params_k%d.pkl' % self.k, 'wb'))





    def evaluate(self, next_batch, running_init, sess, updates, metrics, sample_weights=None):
        all_not_survival = []
        all_events = []
        all_times = []
        all_scales = []
        all_max_hbs = []
        sess.run(running_init)
        for time_batch, event_batch, featidx_batch, featval_batch, minhbs_batch, maxhbs_batch, max_nz_len in next_batch:
            _, _, not_survival, scale_batch, max_hbs_batch  = sess.run(updates, feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'min_headerbids:0': minhbs_batch,
                                             'max_headerbids:0': maxhbs_batch,
                                             'times:0': time_batch,
                                             'events:0': event_batch})
            all_not_survival.extend(not_survival)
            all_events.extend(event_batch)
            all_times.extend(time_batch)
            all_scales.extend(scale_batch)
            all_max_hbs.extend(max_hbs_batch)

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
        return sess.run(metrics), all_not_survival, all_scales, all_max_hbs, all_events, all_times



if __name__ == "__main__":
    with open('output/FeatVec_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvival(
        distribution=Distributions.WeibullDistribution(),
        batch_size=2048,
        num_epochs=20,
        k=80,
        learning_rate=1e-6,
        lambda_linear=0.0,
        lambda_factorized=0.0,
        importance_failure_rate_optimize=0.1,
        importance_optimal_reserve_optimize=5.0
    )

    print('Start training...')
    model.run_graph(num_features,
                    SurvivalData(*pickle.load(open('output/TRAIN_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE),
                    SurvivalData(*pickle.load(open('output/VAL_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE,
                                 only_hb_imp = ONLY_HB_IMP),
                    SurvivalData(*pickle.load(open('output/TEST_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE,
                                 only_hb_imp=ONLY_HB_IMP),
                    sample_weights='time')
