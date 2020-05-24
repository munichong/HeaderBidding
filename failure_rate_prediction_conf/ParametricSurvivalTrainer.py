from time import time as nowtime

import csv
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import log_loss, accuracy_score

from failure_rate_prediction_conf import Distributions
from failure_rate_prediction_conf.DataReader import SurvivalData
from failure_rate_prediction_conf.EvaluationMetrics import c_index
from failure_rate_prediction_conf.ParametricSurvivalModel import ParametricSurvivalModel

ONLY_FREQ_TRAIN = False
ONLY_FREQ_TEST = False

ONLY_HB_IMP = False

MIN_OCCURRENCE = 5


class ParametricSurvivalTrainer:

    def __init__(self, distribution, num_features, batch_size, num_epochs, k, learning_rate=0.001,
                 lambda_linear=0.0, lambda_factorized=0.0,
                 importance_failure_rate_optimize=0.0, importance_optimal_reserve_optimize=0.0):
        self.distribution = distribution
        self.num_features = num_features
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_linear = lambda_linear
        self.lambda_factorized = lambda_factorized
        self.importance_failure_rate_optimize = importance_failure_rate_optimize
        self.importance_optimal_reserve_optimize = importance_optimal_reserve_optimize

    def run_graph(self, train_data, val_data, test_data, sample_weights=None):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 0; when k=0, it is a simple model; Otherwise it is factorized
        :return:
        '''
        model = ParametricSurvivalModel(self.distribution, self.batch_size, self.k, self.num_features, self.lambda_linear, self.lambda_factorized)

        num_total_batches = int(np.ceil(train_data.num_instances / self.batch_size))
        expected_revenue_optimizer = tf.optimizers.Adam(learning_rate=1.0)
        combined_loss_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        for epoch in range(1, self.num_epochs + 1):
            model.reset_metrics()
            num_batch = 0
            for hist_reserve_prices, was_failed, featidx, featval, minhbs, maxhbs, max_nz_len \
                    in train_data.make_sparse_batch(self.batch_size, only_freq=ONLY_FREQ_TRAIN):
                num_batch += 1
                if hist_reserve_prices.shape[0] < self.batch_size:
                    print("Skip the batch whose size is %d < %d" % (hist_reserve_prices.shape[0], self.batch_size))
                    break

                hist_reserve_prices = tf.constant(hist_reserve_prices, dtype=tf.float32)
                was_failed = tf.constant(was_failed, dtype=tf.int32)  # if the historical reserve price was failed to be outbid

                ''' =============== Optimize the optimal reserve price what can maximize the lower bound expected revenue first =============== '''
                prev_loss_expected_revenue_mean = None
                model.initialize_optimal_reserve_prices(featidx.shape[0])
                i = 0
                while True:
                    with tf.GradientTape() as tape1:
                        ''' Compute The Lower Bound Expected Revenue Based on Current Parameters '''
                        loss_expected_revenue_mean = model.loss_lower_bound_expected_revenue(featidx, featval, maxhbs, trainable=True)
                    ''' Optimize Lower Bound Expected Revenue '''
                    grads = tape1.gradient(loss_expected_revenue_mean, [model.optimal_reserve_prices])
                    expected_revenue_optimizer.apply_gradients(grads_and_vars=zip(grads, [model.optimal_reserve_prices]))
                    # expected_revenue_op = expected_revenue_optimizer.minimize(loss=model.loss_expected_revenue_mean, var_list=[model.optimal_reserve_prices])
                    if i % 20 == 0:
                        print('Epoch %d - Batch %d/%d: \tloss_expected_revenue_mean = %.4f' % (epoch, i, num_total_batches, loss_expected_revenue_mean.numpy()))
                        print(model.optimal_reserve_prices.numpy())
                    i += 1
                    if prev_loss_expected_revenue_mean is not None and \
                            (prev_loss_expected_revenue_mean < loss_expected_revenue_mean or
                             0 <= (loss_expected_revenue_mean - prev_loss_expected_revenue_mean) / prev_loss_expected_revenue_mean < 0.0001):
                        print('Epoch %d - Batch %d/%d: \tloss_expected_revenue_mean = %.4f' % (epoch, i, num_total_batches, loss_expected_revenue_mean.numpy()))
                        break
                    prev_loss_expected_revenue_mean = loss_expected_revenue_mean

                with tf.GradientTape() as tape2:
                    ''' =============== Compute the Failure Rate of The Historical Reserve Prices =============== '''
                    hist_failure_proba, hist_failure_bin = model.compute_hist_failure_rate(featidx, featval, hist_reserve_prices)
                    running_failure_event_acc, loss_failure_event = model.loss_hist_failure_rate(hist_failure_proba, hist_failure_bin, hist_reserve_prices, was_failed, sample_weights)

                    ''' =============== Calculate the Loss of Wrong-Side Optimal Revenue =============== '''
                    loss_optimal_reserve = model.loss_optimal_reserve_error(model.optimal_reserve_prices, hist_reserve_prices, hist_failure_bin)  # or was_failed?

                    ''' ============= L2 regularized sum of squares loss function over the embeddings ============= '''
                    loss_complexity = model.loss_complexity()

                    ''' ====================== Combine and Optimize All Losses ======================= '''
                    combined_mean_loss = loss_expected_revenue_mean + \
                                         self.importance_failure_rate_optimize * loss_failure_event + \
                                         self.importance_optimal_reserve_optimize * loss_optimal_reserve + \
                                         loss_complexity

                ''' Optimize Combined Loss '''
                # print(model.trainable_variables)  # TODO: remove optimal_reserve_prices
                grads = tape2.gradient(combined_mean_loss, model.variables)
                combined_loss_optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

                ### gradient clipping
                # combined_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # gradients, variables = zip(*combined_optimizer.compute_gradients(combined_mean_loss, var_list=[embeddings_linear, embeddings_factorized, fm_intercept, shape]))
                # gradients_clipped, _ = tf.clip_by_global_norm(gradients, 100)
                # combined_training_optimizer = combined_optimizer.apply_gradients(zip(gradients_clipped, variables))

                if epoch == 1:
                    print("Epoch %d - Batch %d/%d: combined_mean_loss = %.4f, loss_expected_revenue_mean = %.4f, "
                          "loss_failure_event = %.4f, loss_optimal_reserve = %.4f, loss_complexity = %.4f" %
                          (epoch, num_batch, num_total_batches, combined_mean_loss, loss_expected_revenue_mean,
                           loss_failure_event, loss_optimal_reserve, loss_complexity))

            print()
            print("========== Evaluation at Epoch %d ==========" % epoch)
            # evaluation on validation data
            print('*** On Validation Set:')

            (optimal_reserve_prices_val, mean_loss_expected_revenue_val,mean_loss_failure_event_val,
             running_failure_event_acc_val, mean_loss_optimal_reserve) = self.evaluate(model, val_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST), sample_weights)

            # (loss_val, acc_val), not_survival_val, _, _, events_val, times_val = self.evaluate(
            #     val_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST),
            #     running_vars_initializer, sess,
            #     eval_nodes_update, eval_nodes_metric,
            #     sample_weights)
            # # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_val, acc_val))
            # print("Validation C-Index = %.4f" % c_index(events_val, not_survival_val, times_val))
            #
            # if max_loss_val is None or loss_val < max_loss_val:
            #     print("!!! GET THE LOWEST VAL LOSS !!!")
            #     max_loss_val = loss_val
            #
            #     # evaluation on test data
            #     print('*** On
            #     (loss_test,
            #      acc_test), not_survival_test, scale_test, max_hbs_test, events_test, times_test = self.evaluate(
            #         test_data.make_sparse_batch(only_freq=ONLY_FREQ_TEST),
            #         running_vars_initializer, sess,
            #         eval_nodes_update, eval_nodes_metric,
            #         sample_weights)
            #     # print("TENSORFLOW:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))
            #     print("TEST C-Index = %.4f" % c_index(events_test, not_survival_test, times_test))
            #
            #     # Store prediction results
            #     with open('output/all_predictions_factorized.csv', 'w', newline="\n") as outfile:
            #         csv_writer = csv.writer(outfile)
            #         csv_writer.writerow(('NOT_SURV_PROB', 'EVENTS', 'MAX(RESERVE, REVENUE)', 'MAX_HB', 'SCALE'))
            #         for p, e, t, h, sc in zip(not_survival_test, events_test, times_test, max_hbs_test, scale_test):
            #             csv_writer.writerow((p, e, t, h, sc))
            #     print('All predictions are outputted for error analysis')
            #
            #     # # Store parameters
            #     # params = {'embeddings_linear': embeddings_linear.eval(),
            #     #           'intercept': intercept.eval(),
            #     #           'shape': shape.eval(),
            #     #           'distribution_name': type(self.distribution).__name__}
            #     # if embeddings_factorized is not None:
            #     #     params['embeddings_factorized'] = embeddings_factorized.eval(),
            #     # pickle.dump(params, open('output/params_k%d.pkl' % self.k, 'wb'))

    def evaluate(self, model, next_batch, sample_weights=None):
        optimal_reserve_prices = []
        running_loss_expected_revenue = []
        running_loss_failure_event = []
        running_failure_event_acc = 0
        running_loss_optimal_reserve = []
        for hist_reserve_prices, was_failed, featidx, featval, minhbs, maxhbs, max_nz_len in next_batch:
            # TODO: Need to compute the optimized reserve prices raw iteratively
            loss_expected_revenue_mean = model.loss_lower_bound_expected_revenue(featidx, featval, maxhbs)

            hist_failure_proba, hist_failure_bin = model.compute_hist_failure_rate(featidx, featval, hist_reserve_prices)
            running_failure_event_acc_batch, loss_failure_event = model.loss_hist_failure_rate(hist_failure_proba, hist_failure_bin, hist_reserve_prices, was_failed, sample_weights)

            loss_optimal_reserve = model.loss_optimal_reserve_error(model.optimal_reserve_prices, hist_reserve_prices, hist_failure_bin)  # or was_failed?


            optimal_reserve_prices.extend(model.optimal_reserve_prices.numpy())
            running_loss_expected_revenue.append(loss_expected_revenue_mean.numpy())
            running_failure_event_acc = running_failure_event_acc_batch
            running_loss_failure_event.append(loss_failure_event.numpy())
            running_loss_optimal_reserve.append(loss_optimal_reserve.numpy())

        return optimal_reserve_prices, np.mean(running_loss_expected_revenue), np.mean(running_loss_failure_event), running_failure_event_acc, np.mean(running_loss_optimal_reserve)



if __name__ == "__main__":
    with open('output/FeatVec_adxwon.csv') as f:
        ''' The first line is the total number of unique features '''
        num_features = int(f.readline())

    model = ParametricSurvivalTrainer(
        distribution=Distributions.WeibullDistribution(),
        num_features=num_features,
        batch_size=2048,
        num_epochs=20,
        k=20,
        learning_rate=1e-2,
        lambda_linear=0.0,
        lambda_factorized=0.0,
        importance_failure_rate_optimize=10.0,
        importance_optimal_reserve_optimize=0.00
    )

    print('Start training...')
    model.run_graph(SurvivalData(*pickle.load(open('output/TRAIN_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE),
                    SurvivalData(*pickle.load(open('output/VAL_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE,
                                 only_hb_imp=ONLY_HB_IMP),
                    SurvivalData(*pickle.load(open('output/TEST_SET.p', 'rb')),
                                 min_occurrence=MIN_OCCURRENCE,
                                 only_hb_imp=ONLY_HB_IMP),
                    sample_weights='time')
