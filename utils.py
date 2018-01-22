"""
This module include the LayerNormGRUCell (LNGRU), LNGRU2, and BucketedDataIterator class.

LNGRU and LNGRU2 are variants of the GRU cell with layer normalization. LNGRU applies
layer normalization to the sum of linear transformation of both previous states and inputs,
while LNGRU2 applies layer normalization to the linear transformation of the previous states
and of the inputs separately. When using identity as the activation function, cell states
converge to normal distribution, thus I call the cell "self-normalizing GRU."

BucketedDataIterator takes a list of train sequences and generate mini batches in a way that
improve efficiency of RNN training with input of variable length. It sorts the list of sequences
by lengths and divide the list into K equal buckets. To to draw a batch of M samples, it pick
one of the bucket randomly, and pick M samples randomly from the bucket. The implementation is
from https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import pandas as pd

def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


def linear(input, output_dim, scope='linear', stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('biases', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def layer_norm(tensor, scope=None, epsilon=1e-5, fixed=False, shift_initializer=0.0):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
    if fixed:
        return LN_initial

    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1.0, dtype=tensor.dtype))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(shift_initializer, dtype=tensor.dtype))
    return LN_initial * scale + shift

class LayerNormGRUCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's GRU cell to use layer normalization.
    Note that state_is_tuple is always true
    """
    def __init__(self,
                 num_units,
                 layer_norm=True,
                 keep_prob=1.0,
                 activation=tf.nn.tanh,
                 reuse=None,
                 reset_bias=1.0,  # initialized by default to 1 to increase the influence of h_{t-1} at the beginning
                 update_bias=1.0, # initialized by default to 1 to increase the influence of h_{t-1} at the beginning
                 kernel_initializer=None,
                 seed=None):
        super(LayerNormGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._layer_norm = layer_norm
        self._keep_prob = keep_prob
        self._activation = activation
        self._fixed = activation is None or activation == tf.nn.relu
        self._reset_bias = reset_bias
        self._update_bias = update_bias
        self._kernel_initializer = kernel_initializer
        self._seed = seed

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """ Layer Normalized Gated Recurrent Units (LNGRU) """
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("gates"):
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, False,
                                            kernel_initializer=self._kernel_initializer)
                value = self._gate_linear([inputs, state])
                r, u = tf.split(value, axis=1, num_or_size_splits=2)

                # add layer normalization to each gate
                if self._layer_norm:
                    r = layer_norm(r, shift_initializer=self._reset_bias, scope="reset/")
                    u = layer_norm(u, shift_initializer=self._update_bias, scope="update/")

                r = tf.nn.sigmoid(r)
                u = tf.nn.sigmoid(u)

            with tf.variable_scope("candidate"):
                r_state = r * state
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, False,
                                                 kernel_initializer=self._kernel_initializer)
                c = self._candidate_linear([inputs, r_state])

                # add layer normalization to the candidate
                if self._layer_norm:
                    c = layer_norm(c, fixed=self._fixed)
                    # don't learn scale and shift for relu or identity function

                if self._activation is not None:
                    c = self._activation(c)

                # add dropout layer
                if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
                    c = tf.nn.dropout(c, self._keep_prob, seed=self._seed)

                new_h = u * state + (1 - u) * c
        return new_h, new_h

class LNGRU2(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's GRU cell to use layer normalization.
    Note that state_is_tuple is always true
    """
    def __init__(self,
                 num_units,
                 layer_norm=True,
                 keep_prob=1.0,
                 activation=tf.nn.tanh,
                 reuse=None,
                 reset_bias=1.0,  # initialized by default to 1 to increase the influence of h_{t-1} at the beginning
                 update_bias=1.0, # initialized by default to 1 to increase the influence of h_{t-1} at the beginning
                 kernel_initializer=None,
                 seed=None):
        super(LNGRU2, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._layer_norm = layer_norm
        self._keep_prob = keep_prob
        self._activation = activation
        self._fixed = activation is None or activation == tf.nn.relu
        self._reset_bias = reset_bias
        self._update_bias = update_bias
        self._kernel_initializer = kernel_initializer
        self._seed = seed

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _linear_and_layer_norm(self, inputs, scope, scope2, bias=0.0, fixed=False):
        with tf.variable_scope(scope):
            with tf.variable_scope(scope2):
                _linear = _Linear([inputs], self._num_units, False,
                                  kernel_initializer=self._kernel_initializer)
                linear = _linear([inputs])
                if self._layer_norm:
                    linear_ln = layer_norm(linear, shift_initializer=bias, fixed=fixed)
            return linear_ln

    def __call__(self, inputs, state, scope=None):
        """ Layer Normalized Gated Recurrent Units (LNGRU) """
        with tf.variable_scope(scope or type(self).__name__):
            r_input = self._linear_and_layer_norm(inputs, scope="reset_gate", scope2="input", bias=self._reset_bias)
            r_state = self._linear_and_layer_norm(state, scope="reset_gate", scope2="state", bias=self._reset_bias)
            r = tf.nn.sigmoid(r_input + r_state)

            u_input = self._linear_and_layer_norm(inputs, scope="update_gate", scope2="input", bias=self._reset_bias)
            u_state = self._linear_and_layer_norm(state, scope="update_gate", scope2="state", bias=self._update_bias)
            u = tf.nn.sigmoid(u_input + u_state)

            c_input = self._linear_and_layer_norm(inputs, scope="candidate", scope2="input", fixed=self._fixed)
            c_state = self._linear_and_layer_norm(state, scope="candidate", scope2="state", fixed=self._fixed)
            c = c_input + r * c_state

            if self._activation is not None:
                c = self._activation(c)

            # add dropout layer
            if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
                c = tf.nn.dropout(c, self._keep_prob, seed=self._seed)

            new_h = u * state + (1 - u) * c
        return new_h, new_h

class BucketedDataIterator():
    def __init__(self, df, num_buckets=5):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = len(df) // num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[bucket * self.size : (bucket + 1) * self.size])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()
        self.epochs = 0

    def shuffle(self):
        # sort dataframe by sequence length, but keeps it random within the same buckets
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor + n > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].iloc[self.cursor[i] : self.cursor[i] + n]
        res = res.reset_index(drop=True)
        self.cursor[i] += n

        # pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros((n, maxlen), dtype=np.int32)
        y = np.zeros((n, np.size(res["y"][0])))
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res["X"].values[i]
            y[i] = res["y"].values[i]

        return x, y, res["length"].values

class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
      self._build_bias = build_bias

      if args is None or (nest.is_sequence(args) and not args):
          raise ValueError("`args` must be specified")
      if not nest.is_sequence(args):
          args = [args]
          self._is_sequence = False
      else:
          self._is_sequence = True

      # Calculate the total size of arguments on dimension 1.
      total_arg_size = 0
      shapes = [a.get_shape() for a in args]
      for shape in shapes:
          if shape.ndims != 2:
              raise ValueError("linear is expecting 2D arguments: %s" % shapes)
          if shape[1].value is None:
              raise ValueError("linear expects shape[1] to be provided for shape %s, "
                               "but saw %s" % (shape, shape[1]))
          else:
              total_arg_size += shape[1].value

      dtype = [a.dtype for a in args][0]

      scope = tf.get_variable_scope()
      with tf.variable_scope(scope) as outer_scope:
          self._weights = tf.get_variable(
              "kernel", [total_arg_size, output_size],
              dtype=dtype,
              initializer=kernel_initializer)
          if build_bias:
              with tf.variable_scope(outer_scope) as inner_scope:
                  inner_scope.set_partitioner(None)
                  if bias_initializer is None:
                      bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
                  self._biases = tf.get_variable(
                      "bias", [output_size],
                      dtype=dtype,
                      initializer=bias_initializer)

  def __call__(self, args):
      if not self._is_sequence:
          args = [args]

      if len(args) == 1:
          res = tf.matmul(args[0], self._weights)
      else:
          res = tf.matmul(tf.concat(args, 1), self._weights)
      if self._build_bias:
          res = tf.nnbias_add(res, self._biases)
      return res