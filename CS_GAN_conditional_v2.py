from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
 

from keras.optimizers import RMSprop

import numpy as np

from keras.backend.tensorflow_backend import set_session

import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class CSGAN():
    def __init__(self, mask=[]):
        # Input shape
        self.img_rows = int(256)
        self.img_cols = int(256/256*64)
        self.img_channels = int(3/3*1)
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.sample_rate=64/801
#        self.sample_rate=64/1665
        
        self.inp_rows = int(801*self.sample_rate)
 #       self.inp_rows = int(1665*self.sample_rate)

        self.inp_cols = int(64)
        self.inp_channels = 1
        
        self.inp_shape = (self.inp_rows, self.inp_cols, self.inp_channels)


#        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = False
#        set_session(tf.Session(config=config))

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)


        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
#        patch = int(self.img_rows / 2**4)
        patch = int(self.img_rows / 2**6)
 
        self.disc_patch = (patch, int(patch/1), 1)

        # Number of filters in the first layer of G and D
        self.gf = 64#64
        self.df = 64#64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
#        self.discriminator.compile(loss='mse',
#            optimizer=optimizer,
#            metrics=['accuracy'])
        self.discriminator.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
#        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.generator.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.inp_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model([img_A, img_B], [valid, fake_A])
#        self.combined.compile(loss=['mse', 'mae'],
#                              loss_weights=[1, 100],
#                              optimizer=optimizer)
        
        self.combined.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0,strides=1):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.inp_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
#        d5 = conv2d(d4, self.gf*8)
        d5=d4
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
 #       u2 = deconv2d(u1, d5, self.gf*8)
        u2=u1
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        u8 = UpSampling2D(size=(2,1))(u7)
        u9 = UpSampling2D(size=(2,1))(u8)

        output_img = Conv2D(self.img_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u9)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True,strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.inp_shape)

        # Concatenate image and conditioning image by channels to produce input
#        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
#
#        d1 = d_layer(combined_imgs, self.df, bn=False)
        d1a = d_layer(img_A, self.df, bn=False,strides=(2,1))
        d2a = d_layer(d1a, self.df, bn=False,strides=(2,1))

        d1b = d_layer(img_B, self.df, bn=False,strides=1)

        
        combined_imgs = Concatenate(axis=-1)([d2a, d1b])
        
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, save_interval=50):

        start_time = datetime.datetime.now()
        imgs_A,imgs_B=self.data_loader.load_data(batch_size,first=True,sample_rate=self.sample_rate)

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ----------------------
                #  Train Discriminator
                # ----------------------
    
                # Sample images and their conditioning counterparts
                imgs_A, imgs_B = self.data_loader.load_data(batch_size)
    
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
    
                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)
    
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_A, imgs_B = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, 1-d_loss[0], 100*d_loss[1], 1-g_loss[0]))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

#    def save_imgs(self, epoch):
#        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
#        r, c = 3, 3
#
#        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True,first=True)
#        fake_A = self.generator.predict(imgs_B)
#
#        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
#
#        # Rescale images 0 - 1
#        gen_imgs = 0.5 * gen_imgs + 0.5
#
#        titles = ['Condition', 'Generated', 'Original']
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt])
#                axs[i, j].set_title(titles[i])
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
#        plt.close()
    def save_model(self,path,name):
        self.generator.save_weights(path+name+'_g.w')
        self.discriminator.save_weights(path+name+'_d.w')
        self.combined.save_weights(path+name+'_c.w')
        np.save(path+name+'_mask',self.data_loader.mask)
        
        
    def load_model(self,path,name):
        self.generator.load_weights(path+name+'_g.w')
        self.discriminator.load_weights(path+name+'_d.w')
        self.combined.load_weights(path+name+'_c.w')
        self.data_loader.mask=np.load(path+name+'_mask.npy')
        

    def save_imgs(self, epoch,r=2,c=2):

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=r*c, is_testing=True,first=True)
        org_imgs=imgs_A
        gen_imgs = self.generator.predict(imgs_B)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c*2)
        #        titles = ['Condition', 'Generated', 'Original']

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,2*j+1].imshow(org_imgs[cnt, :,:,0], cmap='gray')
                
                axs[i,2*j].axis('off')
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("pgan/result_%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    
    num_cores = 2
    
#    if GPU:
#        num_GPU = 1
#        num_CPU = 1
#    if CPU:
    
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)
    
    gan = CSGAN()
#    gan.load_model('C:\\Users\\sohei_000\\pgan\\','model2')
    gan.train(epochs=1000, batch_size=32, save_interval=10)
  #  gan.save_model('C:\\Users\\sohei_000\\pgan\\','model22')
