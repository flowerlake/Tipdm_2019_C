import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import BasicConvLSTMCell
from params import *

class G_model_:
  def __init__ (self,scope,height,width,length,batch_size,layer_num_lstm,kernel_size,kernel_num,kernel_size_dec,num_dec_input,num_dec_output,layer_num_cnn,initial_h_0,initial_c_0,initial_h_1,initial_c_1):
    self.scope = scope
    self.height = height
    self.width = width
    self.length =length 
    self.batch_size = batch_size
    self.layer_num_lstm = layer_num_lstm
    self.kernel_size = kernel_size
    self.kernel_num = kernel_num

    self.kernel_size_dec = kernel_size_dec
    self.num_dec_input = num_dec_input
    self.num_dec_output = num_dec_output
    self.layer_num_cnn  = layer_num_cnn
 

    self.initial_h_0 = initial_h_0 
    self.initial_c_0 = initial_c_0  
    self.initial_h_1 = initial_h_1  
    self.initial_c_1 = initial_c_1  
    self.define_graph()

  def define_graph(self):
    with tf.name_scope('input'):
      self.input_features = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, self.length, self.height, self.width,512])

      #self.ground_truth= tf.place_holder(tf.float32, shape=[FLAGS.batch_size, self.length, 224, 224,1])

      #self.initial_h_0 = tf.place_holder(tf.float32,shape=[FLAGS.batch_size, self.height, self.width,512])
      #self.initial_c_0 = tf.place_holder(tf.float32,shape=[FLAGS.batch_size, self.height, self.width,512])
      #self.initial_h_1 = tf.place_holder(tf.float32,shape=[FLAGS.batch_size, self.height, self.width,1])
      #self.initial_c_1 = tf.place_holder(tf.float32,shape=[FLAGS.batch_size, self.height, self.width,1])
    with tf.variable_scope(self.scope) as scope: 
	
      lstms = []
      lstms_state = []
      for layer_id_, kernel_, kernel_num_ in zip(xrange(self.layer_num_lstm),self.kernel_size,self.kernel_num):
	layer_name_encode = 'conv_lstm'+str(layer_id_)+'enc'
        temp_cell= BasicConvLSTMCell.BasicConvLSTMCell([self.height,self.width],[kernel_,kernel_],kernel_num_,layer_name_encode)
	if layer_id_ ==0:
	   lstms_state.append(tf.concat([self.initial_c_0,self.initial_h_0],3))
	else:
	   lstms_state.append(tf.concat([self.initial_c_1,self.initial_h_1],3)) 
      
        lstms.append(temp_cell)
      
      dec_conv_W = []
      dec_conv_b = []	
      for layer_id_,kernel_,input_,output_ in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
        with tf.variable_scope("dec_conv_{}".format(layer_id_)):
          dec_conv_W.append(tf.get_variable("matrix", shape = kernel_+[output_,input_], initializer = tf.random_uniform_initializer(-0.01, 0.01)))
          dec_conv_b.append(tf.get_variable("bias",shape = [output_],initializer=tf.constant_initializer(0.01)))
	  
      input_ = self.input_features[:,0,:,:,:]
      input_,_=lstms[0](input_,lstms_state[0])	
      input_,_=lstms[1](input_,lstms_state[1])       

      for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
	if layer_id_ == self.layer_num_cnn-1:
	  output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])	  
	  input_ = tf.nn.conv2d_transpose(input_, dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + dec_conv_b[layer_id_]
	  input_ = tf.nn.sigmoid(input_)

	else:
	  output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])                            
          input_ = tf.nn.conv2d_transpose(input_, dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + dec_conv_b[layer_id_]
	  input_ = tf.maximum(0.2 * input_, input_)

      scope.reuse_variables()	

      def forward():
        output = []
	for frame_id in xrange(self.length):
	  input_ = self.input_features[:,frame_id,:,:,:]
	  input_,lstms_state[0]=lstms[0](input_,lstms_state[0])
          input_,lstms_state[1]=lstms[1](input_,lstms_state[1])
	  
	  for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
            if layer_id_ == self.layer_num_cnn-1:
              output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
              input_ = tf.nn.conv2d_transpose(input_, dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + dec_conv_b[layer_id_]
              #input_ = tf.nn.sigmoid(input_)
	      output.append(input_)
            else:
              output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
              input_ = tf.nn.conv2d_transpose(input_, dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + dec_conv_b[layer_id_]
              input_ = tf.maximum(0.2 * input_, input_)


        output = tf.stack(output,axis=1)	  
	return output

      self.logits = forward() #note: output is logits need to convert to sigmoid in testing mode
	
	#self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ground_truth, logits= self.logits)
	#temp_op = tf.train.AdamOptimizer(FLAGS.lr)
        #variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                 self.scope)
	#gvs = temp_op.compute_gradients(self.loss,var_list=variable_collection)
        #capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        #self.train_op = temp_op.apply_gradients(capped_gvs)
	#
	#entropy_loss_summary = tf.summary.scalar('entropy_loss',self.loss)
	#self.summary = tf.summary.merge([entropy_loss_summary])

