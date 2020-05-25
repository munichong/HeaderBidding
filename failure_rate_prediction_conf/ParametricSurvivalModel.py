import tensorflow as tf

class ParametricSurvivalModel(tf.keras.Model):
    def __init__(self, distribution, batch_size, k, num_features, lambda_linear, lambda_factorized):
        super(ParametricSurvivalModel, self).__init__()

        self.distribution = distribution
        self.batch_size = batch_size
        self.k = k
        self.num_features = num_features
        self.lambda_linear = lambda_linear
        self.lambda_factorized = lambda_factorized

        self.dist_shape = tf.Variable(name='dist_shape', initial_value=0.2, trainable=True)
        self._initialize_fm_weights()
        self.failure_event_acc = tf.metrics.BinaryAccuracy()

    def reset_metrics(self):
        self.failure_event_acc.reset_states()

    def initialize_optimal_reserve_prices(self, size, trainable=True):
        self.optimal_reserve_prices = tf.Variable(name='optimal_reserve_prices',
                                                  initial_value=tf.random.truncated_normal(shape=[size], mean=5.0, stddev=1e-1),
                                                  constraint=tf.keras.constraints.NonNeg(),
                                                  validate_shape=False, trainable=trainable)

    def _initialize_fm_weights(self, trainable=True):
        # shape: (batch_size, max_nonzero_len)
        self.weights_linear = tf.Variable(name='weights_linear', initial_value=tf.random.truncated_normal(shape=(self.num_features,), mean=0.0, stddev=1e-5), trainable=trainable)
        self.weights_factorized = None
        if self.k > 0:
            # shape: (batch_size, max_nonzero_len, k)
            self.weights_factorized = tf.Variable(name='weights_factorized', initial_value=tf.random.truncated_normal(shape=(self.num_features, self.k), mean=0.0, stddev=1e-5), trainable=trainable)
        self.fm_intercept = tf.Variable(name='fm_intercept', initial_value=1e-5, trainable=trainable)

    def _linear_function(self, weights_linear):
        return tf.reduce_sum(weights_linear, axis=-1) + self.fm_intercept

    def _factorization_machines(self, weights_factorized):
        dot_product_res = tf.matmul(weights_factorized, tf.transpose(weights_factorized, perm=[0, 2, 1]))
        element_product_res = weights_factorized * weights_factorized
        pairs_mulsum = tf.reduce_sum(tf.multiply(0.5, tf.reduce_sum(dot_product_res, axis=2) - tf.reduce_sum(element_product_res, axis=2)),  axis=-1)
        return pairs_mulsum

    def _compute_scale(self, feature_indice, feature_values):
        filtered_weights_linear = tf.gather(self.weights_linear, feature_indice, axis=0) * feature_values
        scales = self._linear_function(filtered_weights_linear)  # this scale = intercept + weights_linear * feat_values

        filtered_weights_factorized = None
        if self.k > 0:
            filtered_weights_factorized = tf.gather(self.weights_factorized, feature_indice) * \
                                          tf.tile(tf.expand_dims(feature_values, axis=-1), [1, 1, 1])
            factorized_term = self._factorization_machines(filtered_weights_factorized)
            scales += factorized_term

        scales = tf.math.softplus(scales)
        return scales

    def compute_hist_failure_rate(self, featidx, featval, hist_reserve_prices):
        scales = self._compute_scale(featidx, featval)

        ''' ================= Calculate Historical Failure Rate ================== '''
        hist_failure_proba, hist_failure_bin = self._compute_failure_rate(hist_reserve_prices, scales)
        return hist_failure_proba, hist_failure_bin

    def loss_hist_failure_rate(self, hist_failure_proba, hist_failure_bin, hist_reserve_prices, was_failed, sample_weights=None):
        ''' =========== Calculate The Loss of Historical Failure Rate Prediction =========== '''
        batch_failure_event_logloss = None
        if not sample_weights:
            _ = self.failure_event_acc.update_state(y_true=was_failed, y_pred=hist_failure_bin)
            batch_failure_event_logloss = tf.losses.BinaryCrossentropy()(y_true=was_failed, y_pred=hist_failure_proba)
        elif sample_weights == 'time':
            _ = self.failure_event_acc.update_state(y_true=tf.expand_dims(was_failed, 1),
                                                    y_pred=tf.expand_dims(hist_failure_bin, 1),
                                                    sample_weight=hist_reserve_prices)
            batch_failure_event_logloss = tf.losses.BinaryCrossentropy()(y_true=tf.expand_dims(was_failed, 1),
                                                                         y_pred=tf.expand_dims(hist_failure_proba, 1),
                                                                         sample_weight=hist_reserve_prices)
        running_failure_event_acc = self.failure_event_acc.result()
        return running_failure_event_acc, batch_failure_event_logloss

    def _compute_failure_rate(self, hist_res, scales):
        '''
        if event == 0, right-censoring
        if event == 1, left-censoring
        '''
        not_survival_proba = self.distribution.left_censoring(hist_res, scales, self.dist_shape)  # the left area
        not_survival_bin = tf.where(tf.math.greater_equal(not_survival_proba, 0.5), tf.ones(tf.shape(not_survival_proba)), tf.zeros(tf.shape(not_survival_proba)))
        return not_survival_proba, not_survival_bin

    def _compute_lower_bound_expected_revenue(self, opt_reserves, scales, max_hbs):
        optimal_failure_proba, _ = self._compute_failure_rate(opt_reserves, scales)
        return (1 - optimal_failure_proba) * opt_reserves + optimal_failure_proba * max_hbs

    def loss_lower_bound_expected_revenue(self, featidx, featval, max_hbs, trainable=True):
        ''' ================= Calculate Lower Bound Expected Revenue ================== '''
        scales = self._compute_scale(featidx, featval)
        lower_bound_expected_revenue = self._compute_lower_bound_expected_revenue(self.optimal_reserve_prices, scales, max_hbs)
        lower_bound_expected_revenue = tf.where(tf.math.is_nan(lower_bound_expected_revenue), tf.zeros(tf.shape(lower_bound_expected_revenue)), lower_bound_expected_revenue)
        loss_expected_revenue_mean = -1 * tf.reduce_mean(lower_bound_expected_revenue)
        return loss_expected_revenue_mean

    def _wrong_side_large_penalty(self, larger_reserves, less_reserves):
        return tf.math.square(larger_reserves - less_reserves)

    def _right_side_small_penalty(self, larger_reserves, less_reserves):
        return tf.zeros(tf.shape(larger_reserves))

    def loss_optimal_reserve_error(self, opt_reserves, hist_reserves, failure):
        loss1 = tf.where(tf.logical_and(tf.equal(failure, 1), tf.math.greater_equal(opt_reserves, hist_reserves)),
                         self._wrong_side_large_penalty(opt_reserves, hist_reserves),
                         tf.zeros(tf.shape(failure)))
        loss2 = tf.where(tf.logical_and(tf.equal(failure, 1), tf.less(opt_reserves, hist_reserves)),
                         self._right_side_small_penalty(hist_reserves, opt_reserves),
                         tf.zeros(tf.shape(failure)))
        loss3 = tf.where(tf.logical_and(tf.equal(failure, 0), tf.less_equal(opt_reserves, hist_reserves)),
                         self._wrong_side_large_penalty(hist_reserves, opt_reserves),
                         tf.zeros(tf.shape(failure)))
        loss4 = tf.where(tf.logical_and(tf.equal(failure, 0), tf.greater(opt_reserves, hist_reserves)),
                         self._right_side_small_penalty(opt_reserves, hist_reserves),
                         tf.zeros(tf.shape(failure)))
        return tf.reduce_mean(loss1 + loss2 + loss3 + loss4)

    def loss_complexity(self):
        l2_norm = self.lambda_linear * tf.nn.l2_loss(self.weights_linear)
        if self.weights_factorized is not None:
            l2_norm += self.lambda_factorized * tf.nn.l2_loss(self.weights_factorized)
        return l2_norm