import functools

import tensorflow as tf
import inspect

NEURONS_NUM = 100
LAYERS_NUM = 3
LEARNING_RATE = 0.003
DROPOUT_KEEP_PROB = 0.5


def lazy_property(funct):
    """
    Causes the function to act like a property. The function is only evaluated once, when it is accessed for the
    first time. The result is stored and directly returned for later accesses, for the sake of efficiency.
    """
    attribute = '_' + funct.__name__

    @property
    @functools.wraps(funct)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, funct(self))
        return getattr(self, attribute)

    return wrapper


class RNNSequenceClassifier:
    def __init__(self, data, target, dropout_keep_prob, neurons_num=NEURONS_NUM, layers_num=LAYERS_NUM):
        self.data = data
        self.target = target
        self.dropout_keep_prob = dropout_keep_prob
        self._neurons_num = neurons_num
        self._layers_num = layers_num
        # needed to initialize lazy properties
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    self._neurons_num, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    self._neurons_num, forget_bias=0.0, state_is_tuple=True)

        # Recurrent network.
        attn_cell = lstm_cell
        if DROPOUT_KEEP_PROB < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell()] * self._layers_num, state_is_tuple=True)

        # discard the state, since every time we look at a new sequence it becomes irrelevant.
        output, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32, sequence_length=self.length)

        # Select last output.
        last = self._last_relevant(output, self.length)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._neurons_num, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
