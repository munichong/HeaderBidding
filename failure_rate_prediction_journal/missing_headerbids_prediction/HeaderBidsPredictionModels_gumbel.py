import os
import csv
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from time import time as nowtime
from sklearn.metrics import mean_squared_error
from failure_rate_prediction_journal.missing_headerbids_prediction.DataReader import HeaderBiddingData, load_hb_data_all_agents, load_hb_data_one_agent
from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS

MODE = 'one_agent'
INPUT_DIR = '../output'
VECTORS_DIR = os.path.join(INPUT_DIR, 'vectorization')
OUTPUT_PKL_NAME = """prediction_result_gumbel_%s.pkl"""


class HBPredictionModel:

    def __init__(self, batch_size, num_epochs, k, distribution=None, learning_rate=0.001,
                 lambda_linear=0.0, lambda_factorized=0.0, hb_agent=''):
        self.distribution = distribution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_linear = lambda_linear
        self.lambda_factorized = lambda_factorized
        self.hb_agent_name = hb_agent

    def linear_function(self, weights_linear, intercept):
        return tf.reduce_sum(weights_linear, axis=-1) + intercept

    def factorization_machines(self, weights_factorized):
        dot_product_res = tf.matmul(weights_factorized, tf.transpose(weights_factorized, perm=[0,2,1]))
        element_product_res = weights_factorized * weights_factorized
        pairs_mulsum = tf.reduce_sum(tf.multiply(0.5, tf.reduce_sum(dot_product_res, axis=2)
                                        - tf.reduce_sum(element_product_res, axis=2)),
                            axis=-1)
        return pairs_mulsum

    def gumbelPDF(self, x, mu, scale):
        z = (x - mu) / scale
        return z + tf.exp(-z)

    def run_graph(self, train_data, val_data, test_data, early_stop=False, bad_epoch_tol=0, verbose=True):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 0; when k=0, it is a simple model; Otherwise it is factorized
        :return:
        '''
        tf.reset_default_graph()

        num_features = train_data.num_features()

        # INPUTs
        feature_indice = tf.placeholder(tf.int32, name='feature_indice')
        feature_values = tf.placeholder(tf.float32, name='feature_values')
        header_bids_true = tf.placeholder(tf.float32, name='header_bids')

        # shape: (batch_size, max_nonzero_len)
        embeddings_linear = tf.Variable(tf.truncated_normal(shape=(num_features,), mean=0.0, stddev=1e-5))
        filtered_embeddings_linear = tf.nn.embedding_lookup(embeddings_linear, feature_indice) * feature_values
        intercept = tf.Variable(1e-5)
        location = self.linear_function(filtered_embeddings_linear, intercept)

        embeddings_factorized = None
        filtered_embeddings_factorized = None
        if self.k > 0:
            # shape: (batch_size, max_nonzero_len, k)
            embeddings_factorized = tf.Variable(tf.truncated_normal(shape=(num_features, self.k), mean=0.0, stddev=1e-5))
            filtered_embeddings_factorized = tf.nn.embedding_lookup(embeddings_factorized, feature_indice) * \
                                      tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])
            factorized_term = self.factorization_machines(filtered_embeddings_factorized)
            location += factorized_term

        scale = tf.Variable(1.0)
        positive_scale = tf.square(scale) + 1e-6
        log_prob = self.gumbelPDF(header_bids_true, location, positive_scale)
        neg_log_likelihood = tf.reduce_sum(log_prob)

        header_bids_pred = location \
                           # + positive_scale * tf.constant(0.5772)

        batch_loss = tf.losses.mean_squared_error(labels=header_bids_true,
                                                      predictions=header_bids_pred,
                                                      reduction = tf.losses.Reduction.MEAN)
        running_loss, loss_update = tf.metrics.mean(batch_loss)


        # L2 regularized sum of squares loss function over the embeddings
        l2_norm = self.lambda_linear * tf.nn.l2_loss(filtered_embeddings_linear)
        if embeddings_factorized is not None:
            l2_norm += self.lambda_factorized * tf.nn.l2_loss(filtered_embeddings_factorized)

        loss_mean = batch_loss

        training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(neg_log_likelihood + l2_norm)

        ### gradient clipping
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # gradients, variables = zip(*optimizer.compute_gradients(loss_mean))
        # gradients_clipped, _ = tf.clip_by_global_norm(gradients, 5.0)
        # training_op = optimizer.apply_gradients(zip(gradients_clipped, variables))


        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


        with tf.Session() as sess:
            init.run()

            max_loss_val = None
            current_bad_epochs = 0
            num_total_batches = int(np.ceil(train_data.num_instances() / self.batch_size))
            for epoch in range(1, self.num_epochs + 1):
                sess.run(running_vars_initializer)
                ''' model training '''
                num_batch = 0
                start = nowtime()
                for hb_batch, featidx_batch, featval_batch in train_data.make_sparse_batch(self.batch_size):
                    num_batch += 1

                    # print(hb_batch)
                    # print(featidx_batch)
                    # print(featval_batch)

                    _, loss_batch, hb_pred, hb_true, pos_scale = sess.run([training_op, loss_mean, header_bids_pred,
                                                                header_bids_true, positive_scale],
                                             feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'header_bids:0': hb_batch})

                    # print(pos_scale)
                    # print(hb_pred)

                    if verbose and epoch == 1:
                        print("Epoch %d - Batch %d/%d: batch loss = %.4f" %
                              (epoch, num_batch, num_total_batches, loss_batch))
                        print("\t\t\t\ttime: %.4fs" % (nowtime() - start))
                        start = nowtime()

                # evaluation on training data
                eval_nodes_update = [loss_update, neg_log_likelihood, header_bids_pred]
                eval_nodes_metric = [running_loss]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                print('*** On Training Set:')
                [loss_train], _, _ = self.evaluate(train_data.make_sparse_batch(),
                                                                 running_vars_initializer, sess,
                                                                 eval_nodes_update, eval_nodes_metric,
                                                                 )
                print("TENSORFLOW:\tMSE = %.6f" % loss_train)

                # evaluation on validation data
                print('*** On Validation Set:')
                [loss_val], hb_pred_val, hb_true_val = self.evaluate(val_data.make_sparse_batch(),
                                                           running_vars_initializer, sess,
                                                           eval_nodes_update, eval_nodes_metric,
                                                           )
                print("TENSORFLOW:\tMSE = %.6f" % loss_val)

                # print(loss_val, max_loss_val)
                if max_loss_val is None or loss_val < max_loss_val:
                    current_bad_epochs = 0
                    print("!!! GET THE LOWEST VAL LOSS !!!")
                    max_loss_val = loss_val

                    # evaluation on test data
                    print('*** On Test Set:')
                    [loss_test], hb_pred_test, hb_true_test = self.evaluate(test_data.make_sparse_batch(),
                                                                            running_vars_initializer, sess,
                                                                            eval_nodes_update, eval_nodes_metric,
                                                                            )
                    print("TENSORFLOW:\tMSE = %.6f" % loss_test)

                    prediction_result = pd.DataFrame(
                        {'y_pred': hb_pred_test,
                         'y_true': hb_true_test
                         })
                    prediction_result.to_pickle(os.path.join(INPUT_DIR, OUTPUT_PKL_NAME % self.hb_agent_name))

                elif early_stop:
                    current_bad_epochs += 1
                    if current_bad_epochs == bad_epoch_tol:
                        break


    def evaluate(self, next_batch, running_init, sess, updates, metrics):
        all_hb_pred = []
        all_hb_true = []
        total_nlog_like = 0
        sess.run(running_init)
        for hb_batch, featidx_batch, featval_batch in next_batch:
            _, nlog_like, hb_pred  = sess.run(updates, feed_dict={
                                             'feature_indice:0': featidx_batch,
                                             'feature_values:0': featval_batch,
                                             'header_bids:0': hb_batch})
            all_hb_pred.extend(hb_pred)
            all_hb_true.extend(hb_batch)
            total_nlog_like += nlog_like

        all_hb_pred = np.array(all_hb_pred, dtype=np.float32)
        all_hb_true = np.array(all_hb_true, dtype=np.float32)

        # print(all_hb_pred)
        # print(all_hb_true)
        print("Negative Log-Likelihood = %.4f" % total_nlog_like)
        print("SKLEARN:\tMSE = %.6f" % (mean_squared_error(all_hb_true, all_hb_pred)))
        return sess.run(metrics), all_hb_pred, all_hb_true



if __name__ == "__main__":
    if MODE == 'all_agents':
        hb_data_train = HeaderBiddingData()
        hb_data_val = HeaderBiddingData()
        hb_data_test = HeaderBiddingData()
        for i, hb_agent_name in enumerate(HEADER_BIDDING_KEYS):
            print("HB AGENT (%d/%d) %s:" % (i + 1, len(HEADER_BIDDING_KEYS), hb_agent_name))
            hb_data_train.add_data(*load_hb_data_all_agents(VECTORS_DIR, hb_agent_name, 'train'))
            hb_data_val.add_data(*load_hb_data_all_agents(VECTORS_DIR, hb_agent_name, 'val'))
            hb_data_test.add_data(*load_hb_data_all_agents(VECTORS_DIR, hb_agent_name, 'test'))


        print('Building model...')
        model = HBPredictionModel(batch_size=512,
                                  num_epochs=30,
                                  k=20,
                                  learning_rate=1e-4,
                                  lambda_linear=0.0,
                                  lambda_factorized=0.0,
                                  hb_agent=MODE)

        print('Start training...')
        model.run_graph(hb_data_train,
                        hb_data_val,
                        hb_data_test)


    elif MODE == 'one_agent':
        for i, hb_agent_name in enumerate(HEADER_BIDDING_KEYS):
            print("\nHB AGENT (%d/%d) %s:" % (i + 1, len(HEADER_BIDDING_KEYS), hb_agent_name))
            hb_data_train = HeaderBiddingData()
            hb_data_val = HeaderBiddingData()
            hb_data_test = HeaderBiddingData()


            hb_data_train.add_data(*load_hb_data_one_agent(VECTORS_DIR, hb_agent_name, 'train'))
            hb_data_val.add_data(*load_hb_data_one_agent(VECTORS_DIR, hb_agent_name, 'val'))
            hb_data_test.add_data(*load_hb_data_one_agent(VECTORS_DIR, hb_agent_name, 'test'))

            print('Building model...')
            model = HBPredictionModel(batch_size=512,
                                      num_epochs=50,
                                      k=20,
                                      learning_rate=1e-3,
                                      lambda_linear=0.0,
                                      lambda_factorized=0.0,
                                      hb_agent=hb_agent_name)

            print('Start training...')
            model.run_graph(hb_data_train,
                            hb_data_val,
                            hb_data_test,
                            early_stop=True,
                            bad_epoch_tol=3,
                            verbose=False)
        y_pred = []
        y_true = []
        for i, hb_agent_name in enumerate(HEADER_BIDDING_KEYS):
            pred_res = pd.read_pickle(os.path.join(INPUT_DIR, OUTPUT_PKL_NAME % hb_agent_name))
            y_pred.extend(pred_res['y_pred'])
            y_true.extend(pred_res['y_true'])

        print("\n###### FINAL EVALUATION RESULT ######")
        print("SKLEARN:\tMSE = %.6f" % (mean_squared_error(y_true, y_pred)))


