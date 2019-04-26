import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import BasicConvLSTMCell
from VGG_model import *
from data_handler import *
from params import * 
from G_model import *
import os 
import time

#data_dict =np.load('../model/params.npy').item()

def train():
  sess1 = tf.Session()
  data_handler = data_handler_(sess1,batch_size=FLAGS.batch_size) # get batch data

  with tf.Graph().as_default():
    kernel_size_dec = []
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])

    #input_img = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,224,224,3])
    #mask_0 =  tf.placeholder(tf.float32, shape=[FLAGS.batch_size,224,224,64]) # 64: number of channel of conv1
    #output_feature_0 = feature_extract(input_img,mask_0)

    #mask_1 =  tf.placeholder(tf.float32, shape=[FLAGS.batch_size,224,224,64]) # 64: number of channel of conv1
    #output_feature_1 = feature_extract(input_img,mask_1)
    output_feature_0 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])
    output_feature_1 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])
    ground_truth = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.length,224,224,1])
    
    with tf.variable_scope('trainable_params') as scope:

    
      with tf.variable_scope('initial_0'):
        c_matrix_0 = tf.get_variable("matrix_c", shape = [3,3,512,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))      	
        c_bias_0 = tf.get_variable("bias_c",shape = [128],initializer=tf.constant_initializer(0.01))
        c_0 = tf.nn.conv2d(output_feature_0,c_matrix_0,strides=[1,1,1,1], padding='SAME') + c_bias_0
	
        h_matrix_0 = tf.get_variable("matrix_h", shape = [3,3,512,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_0 = tf.get_variable("bias_h",shape = [128],initializer=tf.constant_initializer(0.01))
	h_0 = tf.tanh(tf.nn.conv2d(output_feature_0,h_matrix_0,strides=[1,1,1,1], padding='SAME') + h_bias_0)
        
      with tf.variable_scope('initial_1'):
        c_matrix_1 = tf.get_variable("matrix_c", shape = [3,3,512,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        c_bias_1 = tf.get_variable("bias_c",shape = [128],initializer=tf.constant_initializer(0.01))
	c_1 = tf.nn.conv2d(output_feature_1,c_matrix_1,strides=[1,1,1,1], padding='SAME') + c_bias_1

        h_matrix_1 = tf.get_variable("matrix_h", shape = [3,3,512,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_1 = tf.get_variable("bias_h",shape = [128],initializer=tf.constant_initializer(0.01))
	h_1 = tf.tanh(tf.nn.conv2d(output_feature_1,h_matrix_1,strides=[1,1,1,1], padding='SAME') + h_bias_1)

      G_model = G_model_(scope="G_model",height=14,width=14,length=FLAGS.length,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[128,128],kernel_size_dec=kernel_size_dec,num_dec_input=[128,64,32,16],num_dec_output=[64,32,16,1],layer_num_cnn =4,initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1) 
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth, logits= G_model.logits))

      temp_op = tf.train.AdamOptimizer(FLAGS.lr)
      variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params")
      gvs = temp_op.compute_gradients(loss,var_list=variable_collection)
      capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
      train_op = temp_op.apply_gradients(capped_gvs)
      scope.reuse_variables()
 
      entropy_loss_summary = tf.summary.scalar('entropy_loss',loss)
      summary = tf.summary.merge([entropy_loss_summary])

    sess = tf.Session()
    init = tf.initialize_all_variables()

    sess.run(init)

    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
    """saver """
    saver = tf.train.Saver(tf.all_variables())
    saver_load=tf.train.Saver(tf.all_variables())
    saver_load.restore(sess,"/scratch/ys1297/LSTM_tracking/source/checkpoints/test/model.ckpt-6900")

    for step in xrange(FLAGS.max_step):
      t= time.time()
      feature_0_mask,feature_0_no_mask, all_features, masks = data_handler.GetBatch() # first frame masked feature,all feature from all frames, masks for all frames,       mask_feature:(batch_size,14,14,512), all_features:(batch_size,length,14,14,512),  masks:(batch_size,length,14,14,512)
      G_feed_dict = {output_feature_0: feature_0_no_mask, output_feature_1: feature_0_mask, ground_truth:masks,G_model.input_features:all_features}
      g_summary,_,g_loss = sess.run([summary,train_op,loss],feed_dict= G_feed_dict)
      summary_writer.add_summary(g_summary, step)
      elapsed = time.time() - t

      print("time per batch is " + str(elapsed))
      print(step)

      if step %100==0:
	checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

	
if __name__ == '__main__':
  tf.app.run()	
