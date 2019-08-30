#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

class MyGRUCell(RNNCell):
    """
    Your own basic GRUCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow GRUCell source code. If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow GRUCell, but with your own language.
    """

    def __init__(self, num_units, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the GRU cell.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyGRUCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        self.num_units=num_units
        output = 1
        self.activation = math_ops.tanh
        
        with tf.name_scope ('GRUcell_Par'):
            self.W_tanh = tf.Variable(tf.random_normal([num_units+output,num_units]),name = 'W_tanh')
            self.W_sig = tf.Variable(tf.random_normal([num_units+output,num_units]),name = 'W_sig')
            self.b_tanh = tf.Variable(tf.ones([output]),name = 'b_tanh')
            self.b_sig = tf.Variable(tf.ones([output]),name = 'b_sig')


    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        state_size = self.num_units
        return state_size

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        output_size = self.num_units
        return output_size

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow GRUCell has a lot of other features. But we will not try them. Focus on the
        very basic GRU functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In GRU there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (new_state, new_state). For details check TensorFlow GRUCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        #used : Andrew NG's https://www.youtube.com/watch?v=PjMcA_NlB_8&t=245s
        conc_inp = array_ops.concat([inputs, state], 1)
       
        inputs_tanh = math_ops.matmul(conc_inp, self.W_tanh)
        inputs_tanh = nn_ops.bias_add(inputs, self.b_tanh)
        
        inputs_sig = math_ops.matmul(conc_inp, self.W_sig)
        inputs_sig = nn_ops.bias_add(inputs, self.b_sig)
        
        probability = math_ops.sigmoid(inputs_sig)
        candidate = self.activation(inputs_tanh)
               
        new_state = probability * state + (1 - probability) * candidate
        return new_state, new_state
        