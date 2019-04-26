import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
from VGG_model import *
import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from params import *
import cv2
class data_handler_(object):
  def __init__(self,sess,batch_size=1):
    self.batch_size = batch_size
    self.length = FLAGS.length
    self.sess = sess
    #self.data_file_ = h5py.File('../data/butterfly.h5','r')
    self.list_files = [item[0:-1] for item in open('../data/train_files.txt','r').readlines()]
    #self.list_files = [item[0:-1] for item in open('../data/test_files.txt','r').readlines()]
    self.list_len = len(self.list_files) 
    self.data_file_ = h5py.File('../data/'+self.list_files[0],'r') 
    #self.data_file_ =h5py.File('../data/ball1.h5','r')
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    np.random.seed(100)
    
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)
    self.id_ = 0 # every hdf5 ids
    self.file_id = 0 # all file ids
    

    self.input_img = tf.placeholder(tf.float32, shape=[self.batch_size,224,224,3])
    self.mask =  tf.placeholder(tf.float32, shape=[self.batch_size,224,224,64]) # 64: number of channel of conv1
    self.output_feature =  feature_extract(self.input_img,self.mask)

    
  def GetBatch(self,verbose=False):

    feature_0_mask = np.zeros((self.batch_size,14,14,512))
    feature_0_no_mask = np.zeros((self.batch_size,14,14,512))
    all_features = np.zeros((self.batch_size,self.length,14,14,512))
    all_masks = np.zeros((self.batch_size,self.length,224,224,1))
    
    input_img = np.zeros((self.batch_size,224,224,3))
    mask = np.zeros((self.batch_size,224,224,64))
    all_images = np.zeros((self.batch_size,self.length,224,224,3))
    for i in range(self.batch_size):
      first_frame = self.data_file_['image'][self.indices_[self.id_]]
      first_mask = self.data_file_['mask'][self.indices_[self.id_]]
    
      first_mask= np.repeat(first_mask,64,axis=2)
	
      input_img[i,:,:,:] =first_frame
      mask[i,:,:,:] = first_mask

      feature_0_no_mask[i,:,:,:] = self.data_file_['feature'][self.indices_[self.id_]]
      all_features[i,:,:,:,:] = self.data_file_['feature'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]
      all_masks[i,:,:,:,:] = self.data_file_['mask'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]

      all_images[i,:,:,:,:] = self.data_file_['image'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]
      self.id_ +=1
    self.data_file_.close()
    self.file_id +=1
    if self.file_id >= self.list_len :
      self.file_id = 0
   
    self.data_file_ = h5py.File('../data/'+self.list_files[self.file_id],'r')
    self.id_ = 0
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)
    
      #if self.id_ >= (self.dataset_size):
      #  self.id_ = 0
      #	 self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
      #  np.random.shuffle(self.indices_)     
   

    feature_0_mask = self.sess.run(self.output_feature,feed_dict = {self.input_img:input_img,self.mask:mask})
    return feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images

  def generate_overlay_image(self,images,mask,file_name):
    for i in range(32):
      plt.subplot(4,8,i+1)
      plt.imshow(images[i,:,:,::-1]/255.0,alpha=0.9)
      plt.imshow(mask[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')
	
