import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import os

from scipy.misc import imread,imresize

import numpy as np

def read_bmode(path,img_shape,idx=-1):
    
    listing = os.listdir(path)
    if(np.max(idx)<0):
        idx=range(np.shape(listing)[0])
    
    l1=list()
    for i1 in idx:
        ifile=listing[i1]
        img=imread(path+"\\"+ifile)
        img=np.pad(img,pad_width=10,mode='constant',constant_values=0)
        img=imresize(img,size=img_shape)#(64,16))
        l1.append(img)  
    l1=np.array(l1)
    
    #l1.resize(l1.shape[0],64,16)
    bout=l1

    return bout

def mix_noise(input_data,betta):
    output=input_data*(1-betta)+np.random.randn(np.prod(input_data.shape)).reshape(input_data.shape)*betta
    return output

def read_rf(path,idx=-1):
    
    listing = os.listdir(path)
    if(np.max(idx)<0):
        idx=range(np.shape(listing)[0])

    l1=list()
    for i1 in idx:
        ifile=listing[i1]
        csv = np.genfromtxt (path+"\\"+ifile, delimiter=",")
        l1.append(csv)  
    l1=np.pad(l1,pad_width=((0,),(10,),(0,)),mode='constant', constant_values=0)    
    return l1

def sample_rf(rf,sample_ratio,mask=None,flag=False,step=-1):
    if step<0:
        step=int(1/sample_ratio)    
    if flag:
        mask=np.zeros(shape=(rf.shape[2],int(rf.shape[2]*sample_ratio)))
        mask1=(np.random.rand(rf.shape[2])<0.5).reshape(rf.shape[2],)
        for i1 in range(int(rf.shape[2]*sample_ratio)):
            mask[:,i1]=np.roll(mask1,i1*step)
    rf_train=np.matmul(rf,mask)
        
    rfout=rf_train.reshape((rf_train.shape[0],rf_train.shape[1],rf_train.shape[2]))
    rfout=np.swapaxes(rfout,1,2)
    rfout = np.expand_dims(rfout, axis=3)

    return rfout,mask



class DataLoader():
    def __init__(self, img_res=(128, 128), mask=[]):
       # self.dataset_name = dataset_name
        self.img_res = img_res
        self.imgs_A = []
        self.imgs_B = []
        self.sample_rate=1
        self.mask=mask

    def load_data(self, batch_size=1, is_testing=False,first=False,sample_rate=0):
#        data_type = "train" if not is_testing else "test"
#
#        if bmode:
#            path = "C:\\Users\\sohei_000\\Desktop\\train1\\a"#glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
#        if not bmode:
#            path= "C:\\Users\\sohei_000\\Desktop\\train1\\b"
#            
#        batch_images = np.random.choice(path, size=batch_size)
        
        if(sample_rate!=0):
            self.sample_rate=sample_rate


#        if first:
        
#        for img_path in batch_images:
#            img = self.imread(img_path)
#
#            h, w, _ = img.shape
#            _w = int(w/2)
#            img_A, img_B = img[:, :_w, :], img[:, _w:, :]
#

        if not is_testing:
            path_bmode="C:\\Users\\sohei_000\\Desktop\\train1\\a"
            path_rf="C:\\Users\\sohei_000\\Desktop\\train1\\b"
        else:
            path_bmode="C:\\Users\\sohei_000\\Desktop\\train1\\a_t"
            path_rf="C:\\Users\\sohei_000\\Desktop\\train1\\b_t"
        
        listing = os.listdir(path_bmode)
        idx = np.random.randint(0, np.shape(listing)[0], batch_size)
        self.imgs_A=read_bmode(path_bmode,self.img_res,idx)
        self.imgs_B=read_rf(path_rf,idx)
                
            # If training => do random flip
#            if not is_testing and np.random.random() < 0.5:
#                img_A = np.fliplr(img_A)
#                img_B = np.fliplr(img_B)

#            imgs_A.append(img_A)
#            imgs_B.append(img_B)
        
#        imgs_A = scipy.misc.imresize(imgs_A, self.img_res)
#        imgs_B = scipy.misc.imresize(imgs_B, self.inp_res)


        self.imgs_A = np.array(self.imgs_A)/127.5 - 1.
    
        self.imgs_B = np.array(self.imgs_B)#/127.5 - 1.
        imgs_A=self.imgs_A
        imgs_B=self.imgs_B
#        else:
            
#            imgs_A=self.imgs_A[idx]
#            imgs_B=self.imgs_B[idx]
            
        imgs_A = np.expand_dims(imgs_A, axis=3)

        [imgs_B,self.mask]=sample_rf(imgs_B,sample_ratio=self.sample_rate,mask=self.mask,flag=self.mask==[])
        #imgs_B=imgs_B.reshape(imgs_B.shape[0],imgs_B.shape[1],imgs_B.shape[2])
#        imgs_B=np.swapaxes(imgs_B,1,3)
 #       imgs_A=np.swapaxes(imgs_A,1,3)
                
        return imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)