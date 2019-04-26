import pdb
import tensorflow as tf
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class ConvRNNCell(object):
  """Abstract object representing an Convolutional RNN cell.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
      filled with zeros
    """
    
    shape = self.shape 
    num_features = self.num_features
    zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2]) 
    #zeros = tf.random_uniform([batch_size, shape[0], shape[1], num_features * 2],minval=-0.01,maxval=0.01)
    return zeros

class BasicConvLSTMCell(ConvRNNCell):
  """Basic Conv LSTM recurrent network cell. The
  """

  def __init__(self, shape, filter_size, num_features, name = None,forget_bias=1.0, input_size=None,
               state_is_tuple=False, activation=tf.nn.tanh):
    """Initialize the basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      num_features: int thats the depth of the cell 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    #if not state_is_tuple:
      #logging.warn("%s: Using a concatenated state is slower and will soon be "
      #             "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self.shape = shape 
    self.filter_size = filter_size
    self.num_features = num_features 
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

    self.name = name
  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or self.name):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        #c, h = tf.split(3, 2, state) # version 0.1
        c,h = tf.split(state,2,3) # version1.0
      concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)
 
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      #i, j, f, o = tf.split(3, 4, concat) # version 0.1
      i,j,f,o = tf.split(concat,4,3) #version 1.0
      """ yilin modification """      
      with tf.name_scope(scope or 'c_i'):
        weights_i = tf.get_variable("Matrix_i",c.get_shape().as_list()[1:4],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
        biases_i = tf.get_variable("biases_i",[c.get_shape().as_list()[3]],initializer=tf.constant_initializer(0))   
        c_i_temp  = tf.multiply(c, weights_i) + biases_i 
      
       
      with tf.name_scope(scope or 'c_f'):
        weights_f = tf.get_variable("Matrix_f",c.get_shape().as_list()[1:4],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
        biases_f = tf.get_variable("biases_f",[c.get_shape().as_list()[3]],initializer=tf.constant_initializer(0))          
        c_f_temp  = tf.multiply(c, weights_f) + biases_f

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias+c_f_temp) + tf.nn.sigmoid(c_i_temp+i) *
               self._activation(j))

      with tf.name_scope(scope or 'c_o'):
        weights_o = tf.get_variable("Matrix_o",c.get_shape().as_list()[1:4],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
        biases_o = tf.get_variable("biases_o",[c.get_shape().as_list()[3]],initializer =tf.constant_initializer(0))          
        c_o_temp  = tf.multiply(c, weights_o) + biases_o

      new_h = self._activation(new_c) * tf.nn.sigmoid(o+c_o_temp)
      
      """old  """
      #new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
      #         self._activation(j))
      #new_h = self._activation(new_c) * tf.nn.sigmoid(o)      

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h],3)
      return new_h, new_state

def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
  """convolution:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 4D Tensor with shape [batch h w num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]
  
  #pdb.set_trace()
  # Now the computation.
  with tf.variable_scope(scope or "Conv"):
    matrix = tf.get_variable(
		"Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32))
    #variable_summaries(matrix)
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(tf.concat( args,3), matrix, strides=[1, 1, 1, 1], padding='SAME')
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [num_features],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term


if __name__ == '__main__':
  with tf.variable_scope('conv_lstm_1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell([10,9], [3,3], 128)
    #cell(0,0)
    new_state = cell.zero_state(40, tf.float32)
  with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell_1 = BasicConvLSTMCell([10,9], [3,3], 128)
    new_state_1 = cell_1.zero_state(40, tf.float32)
  pdb.set_trace()
  assert cell is cell_1
 
