# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 10:53:17 2018

@author: Aditi Panda
"""

import tensorflow as tf
import numpy as np
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time
import os
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from net import *
from custom_vgg16 import *

tf.reset_default_graph()

class WIN5(object):
    def __init__(self, sess, patch_size=32, batch_size=64,
                 output_size=32, input_c_dim=1, output_c_dim=1, depth=3,
                 clip_b=0.025, lr=0.001,  epoch=10,
                 ckpt_dir='./checkpoint_new', sample_dir='./sample_new_after-4th-epoch',
                 test_save_dir='./test_new', sigma=10, lambda_tv=10e-4, lambda_feat=1e0,
                 dataset='Clean_Training', testset='Clean_Testing_15', evalset = 'eval', load_flag=False, initial_epoch=0):
#        tf.reset_default_graph()
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.depth = depth
        self.sigma = sigma
        self.clip_b = clip_b
        self.numEpoch = epoch
        self.ckpt_dir = ckpt_dir
        self.trainset = dataset
        self.testset = testset
        self.evalset = eval
        self.sample_dir = sample_dir
        self.test_save_dir = test_save_dir
        self.epoch = epoch
        self.save_every_epoch = 1
        self.lambda_tv = lambda_tv
        self.lambda_f = lambda_feat
        #self.eval_every_epoch = 2
        self.load_flag = load_flag
        self.initial_epoch = initial_epoch
        self.abs_epoch_num = self.initial_epoch
#        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.epsilon = 1e-8
        #####################################
        # SGD setting
        self.lr = lr
#        self.weightDecay = 1e-4
#        self.momentum = 0.9
        self.build_model()

    def build_model(self):
#        tf.reset_default_graph()
        # input : [batchsize, patch_size, patch_size, channel]
        self.data_dict = loadWeightsData('./vgg16.npy')
        for key in self.data_dict:
            print(key, self.data_dict[key][0].shape, self.data_dict[key][1].shape)
        # the network structure has to be created in all cases (training for the first time, incremental training, and testing)
        self.create_variables()

        # if load_flag = False, the model is being trained for the first time or tested. If the former is true,
#        parameters like placeholders, learning algo and optimization functions need to be added to a collection,
#        so that they could be easily extracted and used after restoration of the saved model.
        if not self.load_flag:
            print('block 1 of build_model')

#            self.create_variables()
#            self.sess.run(self.init)
            print(self.initial_epoch)
#            tf.add_to_collection('loss_op', self.loss)
##            print(tf.get_collection('loss_op'))
#            tf.add_to_collection('input', self.X)
#            tf.add_to_collection('target', self.X_)
#            tf.add_to_collection('output', self.Y_)
#            tf.add_to_collection('training_step', self.train_step)
            print("[*] Created model successfully...")

        else: # this block is executed when incremental training is carried out. The value of the last executed epoch is found out,
        # and the initial epoch is set accordingly. The training now starts from this value of epoch.
            print('block 2 of build_model')

	    #self.batch_size = 128##decomment this after you change batch size to 128 in main.py
            model_dir = "%s-%s-%s" % (self.trainset,
                                      self.batch_size, self.patch_size)
            checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
            curr_path = os.getcwd()
            os.chdir(checkpoint_dir)

            # Find last executed epoch
            history = list(map(lambda x: int(x.split('-')[1][:-5]), glob('WIN5.model-*.meta')))
            last_epoch = np.max(history)
            # Instantiate saver object using previously saved meta-graph
#            self.saver = tf.train.import_meta_graph('WIN5.model-{}.meta'.format(last_epoch))

            # find out latest version amongst saved models
            self.initial_epoch = last_epoch + 1
            self.abs_epoch_num = self.initial_epoch
            print(self.initial_epoch)
            os.chdir(curr_path)


    def create_variables(self):
# this function creates the network structure, i.e., the layers, the loss function, the optimization algos etc.
        self.X = tf.placeholder(tf.float32, [None, self.depth, self.patch_size, self.patch_size, self.input_c_dim],
                                    name='noisy_image')
        self.X_ = tf.placeholder(tf.float32, [None, self.depth, self.patch_size, self.patch_size, self.input_c_dim],
                                     name='clean_image')
        # layers
        with tf.variable_scope('conv1'):
            layer_1_output = self.layer(self.X, [3, 7, 7, self.input_c_dim, 128])
        with tf.variable_scope('conv2'):
            layer_2_output = self.layer(layer_1_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv3'):
            layer_3_output = self.layer(layer_2_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv4'):
            layer_4_output = self.layer(layer_3_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv5'):
            self.Y = self.layer(layer_4_output, [3, 7, 7, 128, self.output_c_dim], useReLU=False)
        # print(self.Y.get_shape().as_list())

	    # temp_var = tf.reduce_mean(layer_5_output,1)
	    # print(temp_var.get_shape().as_list())
     #    # temp_var = layer_5_output[:,1,:,:,:]
     #    self.Y = temp_var # the predicted noise

        ############################ NEW-LOSSES ##############################################################
            #### reshaping to batchsize-by-64-by-64-by-3
        temp_1 = self.Y[:,:,:,:,0] # network's output/learned mapping/(-predicted noise)--R
        temp_1 = tf.transpose(temp_1, perm=[0,2,3,1])
        temp_2 = self.X[:,:,:,:,0] # noisy image--y
        temp_2 = tf.transpose(temp_2, perm=[0,2,3,1])
        temp_3 = self.X_[:,:,:,:,0] # clean image--x
        temp_3 = tf.transpose(temp_3, perm=[0,2,3,1])
        print(temp_1.shape)

        # Noisy feature
        inputs = temp_2
        vgg_c = custom_Vgg16(inputs, data_dict=self.data_dict)
        feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

            # feature after transformation -- Denoised
        outputs = temp_2 + temp_1 ## noisy+(-predicted noise)=denoised---Skip connection
        vgg = custom_Vgg16(outputs, data_dict=self.data_dict)
        feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

            # compute feature loss
        loss_f = tf.zeros(self.batch_size, tf.float32)
        for f, f_ in zip(feature, feature_):
            loss_f += self.lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])
        print(loss_f.get_shape().as_list())

        # total variation denoising
        loss_tv = self.lambda_tv * self.total_variation_regularization(outputs)
        # print(loss_tv.shape)
        ####################################################################################################3

        # L2 loss
        loss_l2 = (1.0 / self.batch_size) * tf.nn.l2_loss(outputs - temp_3) ### owing to skip connections
        # print(loss_l2.shape)
        self.loss = loss_l2 + loss_tv + loss_f

        print(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss) # No gradient clipping;DnCNN --> DConvELUNN

        tf.summary.scalar('loss', self.loss)

        # create this init op after all variables specified, it helps in initializing all variables of the program (weights and biases)
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(max_to_keep=11) # this will be used for saving and restoring trained models in binary files, i.e., checkpointing

        print('variables created')

    def conv_layer(self, inputdata, weightshape, b_init, stridemode):
        # weights
        W = tf.get_variable('weights', weightshape,
                            initializer=tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
#        print(W.shape)
        b = tf.get_variable('biases', [1, weightshape[-1]], initializer=tf.constant_initializer(b_init))
        # convolutional layer
#        print(d_rate)
        return tf.add(tf.nn.conv3d(inputdata, W, strides=stridemode, padding="SAME"), b)  # SAME with zero padding


    def bn_layer(self, logits, output_dim, b_init=0.0):
        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer=\
                                tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer=\
                               tf.constant_initializer(b_init))
        return batch_normalization(logits, alpha, beta, isCovNet=True)

    def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1, 1], useBN=True, useReLU=True):
        logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
        if useReLU == False:
            output = logits
        else:
            if useBN:
                output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
            else:
                output = tf.nn.relu(logits)
        return output

    def train(self):
        print("Inside Train Function")
        self.sess.run(self.init)# initialize the variables of the program, this has to be done in all cases i.e.,training for the first time, incremental training, and testing
        if self.load_flag:#     load the latest trained model saved
            if self.load(self.ckpt_dir):
                print(" [*] Load SUCCESS (in train)")
            else:
                print(" [!] Load failed...(in train)")
        data_gt = load_data(filepath='./data/Clean_Training/clean_3D_patch_array.npy')
        numBatch = int(data_gt.shape[0] / self.batch_size)
        print(numBatch)

        # create file name and an empty list
        file_part1 = 'training-loss-'
        ext = '.npy'

        print("[*] Start training : ")
        print(datetime.datetime.now())
        start_time = time.time()
        for epoch in range(self.initial_epoch, self.epoch):
             # a list for storing loss values epoch wise

            loss_list = []
            for batch_id in xrange(numBatch):
                batch_images = data_gt[batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :, :, :, :]
                batch_images = np.array(batch_images / 255.0, dtype=np.float32) #normalize the data to 0-1, was omitted by mistake, added on June 12th--35th epoch onwards
                # till 34th epoch normalization wasn't there. It improved results, so another experiment will be performed for training with data norm
                #(and new lr range) today onwards.

                train_images = add_rician_noise(batch_images, self.sigma, self.sess)

                _, loss = self.sess.run([self.train_step, self.loss],\
                                        feed_dict={self.X: train_images, self.X_: batch_images})
                # print(loss.shape)
                loss_list.append(loss)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch, batch_id + 1, numBatch,
                         time.time() - start_time, loss[0]))
            self.save(epoch)
            file_name = file_part1 + str(epoch) + ext
            np.save(file_name, loss_list)

            ## below two lines are commented because no validation
            #if np.mod(epoch, self.eval_every_epoch) == 0:
               # self.evaluate(epoch)
            ##########
        print("[*] Finish training.")
        print(datetime.datetime.now())


    def save(self, epoch):
        # create the name of the folder containing the checkpoints
        model_name = "WIN5.model"
        model_dir = "%s-%s-%s" % (self.trainset,
                                  self.batch_size, self.patch_size)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

        # make the folder if it doesn't already exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        #save using the saver object created earlier
        print("[*] Saving model...")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=epoch)

    def sampler(self, image):
        # set reuse flag to True
        # tf.get_variable_scope().reuse_variables()
        self.X_test = tf.placeholder(tf.float32, image.shape, name='noisy_image_test')
        # layer 1 (adpat to the input image)
        with tf.variable_scope('conv1', reuse=True):
            layer_1_output = self.layer(self.X_test, [3, 7, 7, self.input_c_dim, 128])
        # layer 2 to 5
        with tf.variable_scope('conv2', reuse=True):
            layer_2_output = self.layer(layer_1_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv3', reuse=True):
            layer_3_output = self.layer(layer_2_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv4', reuse=True):
            layer_4_output = self.layer(layer_3_output, [3, 7, 7, 128, 128])
        with tf.variable_scope('conv5', reuse=True):
            layer_5_output = self.layer(layer_4_output, [3, 7, 7, 128, self.output_c_dim], useReLU=False)

        temp_var = tf.reduce_mean(layer_5_output, 1)
	    # print(temp_var.get_shape().as_list())
        # temp_var = layer_5_output[:,1,:,:,:]
        self.Y_test = temp_var


    def load(self, checkpoint_dir):
        '''Load checkpoint file'''
        print("[*] Reading checkpoint...")
        # create the name of the folder containing the checkpoints
        model_dir = "%s-%s-%s" % (self.trainset, self.batch_size, self.patch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)


        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def forward(self, noisy_image):
        # assert noisy_image is range 0-1
        self.sampler(noisy_image)
        return self.sess.run(self.Y_test, feed_dict={self.X_test: noisy_image})

    def test(self):## this is actually evaluate -- name changed to test for making it work

        self.sess.run(self.init)
        if self.load(self.ckpt_dir):
            print(" [*] Load SUCCESS (in test)")
        else:
            print(" [!] Load failed...(in test)")
        ######################test part ends here################
        number_of_slices = 150
        ext = '.npy'
        print("[*] Evaluating...")
        psnr_sum = 0
        print(datetime.datetime.now())
        clean_filename = './data/Clean_Validation/IXI021-Guys-Slices-Patches/eval_3d_image_slice_'

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        for idx in xrange(number_of_slices):
            print(idx)
            clean_path = clean_filename + str(idx+1) + ext
            # print(clean_path)
            clean_3d_slice = load_data(filepath=clean_path)
            clean_3d_slice = np.array(clean_3d_slice / 255.0)

            shape_arr = clean_3d_slice.shape
            clean_3d_slice = np.array(clean_3d_slice).reshape(1, shape_arr[0],
                                     shape_arr[1], shape_arr[2], 1)

            noisy_3d_slice = add_rician_noise(clean_3d_slice, self.sigma, self.sess)

            predicted_noise = self.forward(noisy_3d_slice) ## -noise is the output


            # print(predicted_noise.shape)
            # noisy_slice = noisy_3d_slice[:,1,:,:,:]
            # clean_slice = clean_3d_slice[:,1,:,:,:]

            noisy_slice = np.mean(noisy_3d_slice, axis=1)
            clean_slice = np.mean(clean_3d_slice, axis=1)

            output_clean_image = noisy_slice + predicted_noise

            groundtruth = np.clip(255 * clean_slice, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_slice, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')

#            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_sum += psnr
            print(psnr)
            save_images(groundtruth, noisyimage, outputimage, os.path.join(self.sample_dir, 'validate%d.png' % (idx)))
        avg_psnr = psnr_sum / number_of_slices
        print(avg_psnr)
#
        # file_part1 = 'avg-psnr-eval-'
        # file_name = file_part1 + str(epoch) + ext
        # np.save(file_name, avg_psnr)
        # print("--- Evaluate ---- Average PSNR %.2f ---" % avg_psnr)
        print(datetime.datetime.now())


    def total_variation_regularization(self, x, beta_tv=1):
        assert isinstance(x, tf.Tensor)
        wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
        ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
        tvh = lambda x: conv2d(x, wh, p='SAME')
        tvw = lambda x: conv2d(x, ww, p='SAME')
        dh = tvh(x)
        dw = tvw(x)
        tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta_tv / 2.)
        return tv
