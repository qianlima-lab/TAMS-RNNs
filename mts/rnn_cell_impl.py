from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
import numpy as np

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def assert_like_rnncell(cell_name, cell):

  conditions = [
      hasattr(cell, "output_size"),
      hasattr(cell, "state_size"),
      hasattr(cell, "get_initial_state") or hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing",
      "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


def _concat(prefix, suffix, static=False):
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape


def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)


@tf_export("nn.rnn_cell.RNNCell")
class RNNCell(base_layer.Layer):

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    super(RNNCell, self).__init__(
        trainable=trainable, name=name, dtype=dtype, **kwargs)
    self._is_tf_rnn_cell = True

  def __call__(self, inputs, state, scope=None):
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable  # pylint: disable=protected-access
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      if batch_size is not None:
        if tensor_util.is_tensor(batch_size):
          static_batch_size = tensor_util.constant_value(
              batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(
                  inputs.shape[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(
                inputs.dtype, dtype))

      batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if None in [batch_size, dtype]:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)

  def zero_state(self, batch_size, dtype):
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output


class LayerRNNCell(RNNCell):

  def __call__(self, inputs, state, time_step, scope=None, *args, **kwargs):
    return base_layer.Layer.__call__(self, inputs, state, time_step, scope=scope,
                                     *args, **kwargs)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
@tf_export("nn.rnn_cell.LSTMStateTuple")
class LSTMStateTuple(_LSTMStateTuple):
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

    
@tf_export("nn.rnn_cell.TAMSRNN_CELL")
class TAMSRNN_CELL(LayerRNNCell):

  @deprecated(None, "This class is deprecated, please use tf.nn.rnn_cell.LSTMCell, which supports all the feature "
                    "this cell currently has. Please replace the existing code with tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell').")
  def __init__(self, num_units, periods, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, name=None, dtype=None, **kwargs):
    
    super(TAMSRNN_CELL, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnLSTM for better performance on GPU.", self)

    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self.periods = periods
    self.memory_nums = len(self.periods)
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self.group_size = int(self._num_units / self.memory_nums)
    
    self.mask_temp1 = np.zeros((4 * self.memory_nums, 4 * self._num_units), dtype=np.float32)
    self.mask_temp2 = np.zeros((self.memory_nums * self._num_units, 4 * self._num_units), dtype=np.float32)
    self.mask_temp3 = np.zeros((self.memory_nums, 4 * self._num_units), dtype=np.float32)
    for i in range(4 * self.memory_nums):
        self.mask_temp1[i, i*self.group_size:(i+1)*self.group_size] = 1.0
    for i in range(self.memory_nums):
        for j in range(i+1):
            for k in range(4):
                self.mask_temp2[i*self._num_units+j*self.group_size:i*self._num_units+(j+1)*self.group_size, j*self.group_size+k*self._num_units:(j+1)*self.group_size+k*self._num_units] = 1.0
    for i in range(self.memory_nums):
        for j in range(4):
            self.mask_temp3[i, j*self._num_units : j*self._num_units+(i+1)*self.group_size] = 1.0
            
    if activation: self._activation = activations.get(activation)
    else: self._activation = math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[-1] is None: raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[-1]
    self._kernel_w = self.add_variable("kernel_w", shape=[input_depth, 4 * self._num_units])
    self._kernel_u = self.add_variable("kernel_u", shape=[self._num_units, 4 * self._num_units])
    self._bias_w = self.add_variable("bias_w", shape=[4 * self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel_m = self.add_variable("kernel_m", shape=[input_depth, 4 * self.memory_nums])
    self._kernel_m2 = self.add_variable("kernel_m2", shape=[self._num_units, 4 * self.memory_nums])
    self._bias_m = self.add_variable("bias_m", shape=[4 * self.memory_nums], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self.built = True

  def call(self, inputs, state, time_step):

    group_index = 0
    activate_num = 0
    for k in range(self.memory_nums):
        if time_step % self.periods[k] == 0:
            group_index = k+1
    activate_num = group_index
  
    sigmoid = math_ops.sigmoid
    add = math_ops.add
    multiply = math_ops.multiply
    one = constant_op.constant(1, dtype=dtypes.int32)
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
      
    self.input_w = nn_ops.bias_add(math_ops.matmul(inputs, self._kernel_w), self._bias_w)
    d_w = nn_ops.bias_add( math_ops.matmul( inputs, self._kernel_m ) + math_ops.matmul( h, self._kernel_m2 ), self._bias_m)
    d_w1, d_w2, d_w3, d_w4 = array_ops.split(d_w, num_or_size_splits = 4, axis = -1)
    # Both softmax and sigmoid can be used
    d_w_softmax = array_ops.concat([tf.nn.softmax(d_w1), tf.nn.softmax(d_w2), tf.nn.softmax(d_w3), tf.nn.softmax(d_w4)], 1)
    h_dw = math_ops.matmul(d_w_softmax, self.mask_temp1)
    self.input_h = math_ops.matmul(h, (self._kernel_u * self.mask_temp2[ (activate_num-1) * self._num_units : activate_num * self._num_units, :])) * h_dw 
    self.input_all = self.input_w + self.input_h
     
    i, o, j, f = array_ops.split(value = self.input_all, num_or_size_splits = 4, axis = -1)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))
    
    new_h = self.mask_temp3[activate_num-1, :self._num_units] * new_h + (1 - self.mask_temp3[activate_num-1, :self._num_units]) * h
    new_c = self.mask_temp3[activate_num-1, :self._num_units] * new_c + (1 - self.mask_temp3[activate_num-1, :self._num_units]) * c   
    
    if self._state_is_tuple: new_state = LSTMStateTuple(new_c, new_h)
    else: new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state, d_w_softmax

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(TAMSRNN_CELL, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))