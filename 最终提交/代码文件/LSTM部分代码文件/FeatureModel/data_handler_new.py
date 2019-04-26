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
import time
import cProfile
def crop_image_include_coundary(output,input,center):
  height,width = output.shape[0],output.shape[1]
  min_x = max(0,int(center[0]-height/2))
  min_out_x = max(0,int(height/2-center[0])) 

  min_y = max(0,int(center[1]-width/2))
  min_out_y = max(0,int(width/2-center[1]))

  max_x = min(input.shape[0],int(center[0]+height/2))
  max_out_x = min(height, height+ input.shape[0]- int(center[0]+height/2))
  max_y =  min(input.shape[1],int(center[1]+width/2))
  max_out_y = min(width, width+ input.shape[1]- int(center[1]+width/2))     
  
  try:
    output[min_out_x:max_out_x,min_out_y:max_out_y,:] = input[min_x:max_x,min_y:max_y,:]
  except:
    pdb.set_trace()

  return output
  
class data_handler_(object):
  def __init__(self,sess,batch_size=1):
    self.batch_size = batch_size
    self.length = FLAGS.length
    self.sess = sess
    #self.data_file_ = h5py.File('../data/butterfly.h5','r')
    self.list_files = [item[0:-1] for item in open('../data_origin/train_files.txt','r').readlines()]
    #self.list_files = [item[0:-1] for item in open('../data/test_files.txt','r').readlines()]
    self.list_len = len(self.list_files) 
    self.data_file_ = h5py.File('../data_origin/'+self.list_files[0],'r') 
    #self.data_file_ =h5py.File('../data/ball1.h5','r')
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    np.random.seed(100)
    
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)
    self.id_ = 0 # every hdf5 ids
    self.file_id = 0 # all file ids
    

    self.input_img = tf.placeholder(tf.float32, shape=[1,224,224,3])
    self.mask =  tf.placeholder(tf.float32, shape=[1,224,224,64]) # 64: number of channel of conv1
    self.output_feature =  feature_extract(self.input_img,self.mask)

    
  def GetBatch(self,verbose=False):

    feature_0_mask = np.zeros((self.batch_size,14,14,512))
    feature_0_no_mask = np.zeros((self.batch_size,14,14,512))
    all_features = np.zeros((self.batch_size,self.length,14,14,512))
    all_masks = np.zeros((self.batch_size,self.length,224,224,1))
    
    input_img = np.zeros((self.batch_size,224,224,3))
    #mask = np.zeros((self.batch_size,224,224,1))
    all_images = np.zeros((self.batch_size,self.length,224,224,3))

    for i in range(self.batch_size):
      feature_0_mask[i,:,:,:] = self.data_file_['first_frame_feature'][self.indices_[self.id_]]
      feature_0_no_mask[i,:,:,:] = self.data_file_['feature'][self.indices_[self.id_]]
      all_features[i,:,:,:,:] = self.data_file_['feature'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length] 
      all_masks[i,:,:,:,:] = self.data_file_['mask'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]
      all_images[i,:,:,:,:] = self.data_file_['crop_image'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]
  

    self.data_file_.close()
    self.file_id +=1
    if self.file_id >= self.list_len :
      self.file_id = 0

    self.data_file_ = h5py.File('../data_origin/'+self.list_files[self.file_id],'r')
    self.id_ = 0
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)


    return feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images

  def Get_test_Batch(self,verbose=False):
    for i in range(self.batch_size):
      label = self.data_file_['label'][self.indices_[self.id_]]
      center = [int((label[1]+label[3]+label[5]+label[7])/4.0),int((label[0]+label[2]+label[4]+label[6])/4.0)]
      first_frame_ = self.data_file_['image'][self.indices_[self.id_]]

      list_pos = label
      poly_verts = [(list_pos[0],list_pos[1]),(list_pos[2],list_pos[3]),(list_pos[4],list_pos[5]),(list_pos[6],list_pos[7]),(list_pos[0],list_pos[1])]
      first_mask_ = construct_mask(poly_verts,nx=first_frame_.shape[1],ny=first_frame_.shape[0],nchan=1)

      first_frame = crop_image_include_coundary(np.zeros((224*2,224*2,3)), first_frame_,center)
      first_mask = crop_image_include_coundary(np.zeros((224*2,224*2,1)), first_mask_,center)
      first_frame = cv2.resize(first_frame,(224,224),interpolation=cv2.INTER_LINEAR)
      first_mask =  cv2.resize(first_mask.squeeze(),(224,224),interpolation=cv2.INTER_LINEAR) 
      input_img[i,:,:,:] = first_frame
 
      mask_input=  np.repeat(first_mask.reshape(1,224,224,1),64,axis=3)
      feature_0_mask = self.sess.run(self.output_feature,feed_dict = {self.input_img:first_frame.reshape(1,224,224,3),self.mask:mask_input})
      feature_0_no_mask = self.sess.run(self.output_feature,feed_dict = {self.input_img:first_frame.reshape(1,224,224,3),self.mask:np.ones((1,224,224,64))})
      #feature_0_mask = 0
      #feature_0_no_mask =0	
      for item in range(self.length):
	label = self.data_file_['label'][self.indices_[self.id_]+item+1]
	center = [int((label[1]+label[3]+label[5]+label[7])/4.0),int((label[0]+label[2]+label[4]+label[6])/4.0)]
	frame_ = self.data_file_['image'][self.indices_[self.id_]+item+1]
	list_pos = label
        poly_verts = [(list_pos[0],list_pos[1]),(list_pos[2],list_pos[3]),(list_pos[4],list_pos[5]),(list_pos[6],list_pos[7]),(list_pos[0],list_pos[1])]
	mask_ = construct_mask(poly_verts,nx=frame_.shape[1],ny=frame_.shape[0],nchan=1)
        frame = crop_image_include_coundary(np.zeros((224*2,224*2,3)), frame_,center) 
	mask = crop_image_include_coundary(np.zeros((224*2,224*2,1)), mask_,center)
 	
	frame = cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR)
	mask = cv2.resize(mask.squeeze(),(224,224),interpolation=cv2.INTER_LINEAR)
	
	all_features[i,item,:,:,:] = self.sess.run(self.output_feature,feed_dict={self.input_img:frame.reshape(1,224,224,3),self.mask:np.ones((1,224,224,64))})	
 	all_masks[i,item,:,:,0] = mask 
	all_images[i,item,:,:,:] = frame
     		
	
	
    self.data_file_.close()
    self.file_id +=1
    if self.file_id >= self.list_len :
      self.file_id = 0

    self.data_file_ = h5py.File('../data_origin/'+self.list_files[self.file_id],'r')
    self.id_ = 0
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)	
    return feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images

  def generate_overlay_image(self,images,mask,file_name):
    for i in range(32):
      plt.subplot(4,8,i+1)
      plt.imshow(images[i,:,:,::-1]/255.0,alpha=0.9)
      plt.imshow(mask[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')


if __name__ =="__main__":
   sess = tf.Session()
   data_handler = data_handler_(sess,2)
   for i in range(5): 
     t = time.time() 
     feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images = data_handler.GetBatch()
     print time.time()-t
     #data_handler.generate_overlay_image(all_images[0,:,:,:,:],all_masks[0,:,:,:,:],'test'+str(i))
 
     print i



	
