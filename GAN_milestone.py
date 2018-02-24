from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
from scipy.misc import imread

import matplotlib.pyplot as plt

import sys

import numpy as np

def read_bmode(path):
    
    listing = os.listdir(path)
    
    l1=list()
    for ifile in listing:
        img=imread(path+"\\"+ifile)
        l1.append(img)  
    l1=np.array(l1)
    return l1
def mix_noise(input_data,betta):
    output=input_data*(1-betta)+np.random.randn(np.prod(input_data.shape)).reshape(input_data.shape)*betta
    return output

def read_rf(path,num=0):
    
    listing = os.listdir(path)
    
    l1=list()
    for ifile in listing:
        csv = np.genfromtxt (path+"\\"+ifile, delimiter=",")
        l1.append(csv)  
    l1=np.array(l1)/2+0.5
    return l1

#    def sample_rf(rf,sample_ratio):
#        mask=np.random.rand(rf.size).reshape(rf.shape)<sample_ratio
#        randvals=np.random.rand(np.sum(mask<1))
#        rfout=rf
#        rfout[mask<1]=randvals        
#        return rfout,mask

def sample_rf(rf,sample_ratio,mask=None,flag=False):
    if flag:
        mask=(np.random.rand(rf.shape[2]*int(rf.shape[2]*sample_ratio))<0.5).reshape(rf.shape[2],int(rf.shape[2]*sample_ratio))
    
    #randvals=np.random.rand(np.sum(mask<1))
    rfout=np.matmul(rf,mask)
    rfout1=rfout.reshape((rfout.shape[0],rfout.shape[1]*rfout.shape[2]))
#        rfout[mask<1]=randvals        
    return rfout1,mask




class GAN():
    def __init__(self):
        self.img_rows = 236 
        self.img_cols = 44
        self.channels = 1
        self.sample_rate=0.05
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        
        rf_train=read_rf("C:\\Users\\sohei_000\\Desktop\\train1\\b")
        [rf_train_sampled,mask_sampled]=sample_rf(rf_train,self.sample_rate,mask=None,flag=True)
        self.mask=mask_sampled

        z = Input(shape=(rf_train_sampled.shape[1],))
        #z = Input(shape=(44,801,1,))
 
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

#        noise_shape = (44,801,1,)
        noise_shape = (44*int(801*self.sample_rate),)
        
        model = Sequential()
        
        
#        model.add(Conv2D(32, (3, 3), activation='linear', input_shape=(44,801,1)))
 #       model.add(LeakyReLU(alpha=0.2))
  #      model.add(BatchNormalization(momentum=0.8))

#        model.add(Conv2D(32, (3, 3),strides=(2,2) ,activation='linear'))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(Conv2D(64, (3, 3),strides=(2,2), activation='linear'))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(Conv2D(64, (3, 3),strides=(2,2) ,activation='linear'))
#        model.add(LeakyReLU(alpha=0.2))
#        model.add(BatchNormalization(momentum=0.8))
#        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dense(256, input_shape=noise_shape))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    
    def train(self, epochs, batch_size=128, save_interval=100):

        # Load the dataset
        ###(X_train, _), (_, _) = mnist.load_data()
        X_train=read_bmode("C:\\Users\\sohei_000\\Desktop\\train1\\a")
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        rf_train=read_rf("C:\\Users\\sohei_000\\Desktop\\train1\\b")
        [rf_train_sampled,mask_sampled]=sample_rf(rf_train,self.sample_rate,mask=self.mask)


        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = mix_noise(X_train[idx],0)
            rf_train_noisy=mix_noise(rf_train,0.3)
            [rf_train_sampled,mask_sampled]=sample_rf(rf_train_noisy,self.sample_rate,mask=self.mask)
            noise= mix_noise(rf_train_sampled[idx],0)
#            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

#            noise = np.random.normal(0, 1, (batch_size, 100))
            idx = np.random.randint(0, X_train.shape[0], batch_size)

            [rf_train_sampled,mask_sampled]=sample_rf(rf_train,self.sample_rate,mask=self.mask)
            noise= rf_train_sampled[idx]

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2,2
#        noise = np.random.normal(0, 1, (r * c, 800))
        rf_train=read_rf("C:\\Users\\sohei_000\\Desktop\\train1\\b_t")
        idx = np.random.randint(0, rf_train.shape[0], r*c)
       
        
        [rf_train_sampled,mask_sampled]=sample_rf(rf_train,sample_ratio=self.sample_rate,mask=self.mask)
        noise= rf_train_sampled[idx]
        org_imgs=read_bmode("C:\\Users\\sohei_000\\Desktop\\train1\\a_t")
        org_imgs=org_imgs[idx]
   #     noise= rf_train[1:(r*c)]
#
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c*2)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,2*j+1].imshow(org_imgs[cnt, :,:], cmap='gray')
                
                axs[i,2*j].axis('off')
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("gan/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1000, batch_size=128, save_interval=10)




