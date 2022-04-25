#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:04:29 2020

@author: yunghx_macpro
"""

import tensorflow as TF
from src.networks.fcn_simple_net_tf import fcn_simple_net
import numpy as np


class NetWork:
    def __init__(self, Batch_size=1, ImageSize=160,
                 Model_path='C:/Users/320181892/Documents/python/Hongxu_IGT/src/networks/weights/DSUNetHybrid.ckpt'):
        config = TF.ConfigProto()
        config.gpu_options.allow_growth = True
        self.Batch_size = Batch_size
        self.ImageSize = ImageSize
        self.Model_path = Model_path
        self.keep_prob = TF.placeholder(TF.float32, name="keep_probabilty")
        self.Phase_train = TF.placeholder(TF.bool)
        self.Image = TF.placeholder(TF.float32, shape=[Batch_size, ImageSize, ImageSize, ImageSize, 1],
                                    name="input_image")
        self.Pre1, self.Pre2, self.activation_map_shapes = fcn_simple_net(self.Image, self.keep_prob, self.Phase_train, Batch_size)

        self.sess = TF.Session(config=config)

        print("Setting up Saver...")
        self.saver = TF.train.Saver()
        self.sess.run(TF.global_variables_initializer())

        self.saver.restore(self.sess, self.Model_path)

    def prediction_p(self, Input):
        feed_dict = {self.Image: Input, self.keep_prob: 1.0, self.Phase_train: False}
        prediction1, prediction2, = self.sess.run([self.Pre1, self.Pre2], feed_dict)
        return prediction1, prediction2

    def predict(self, image):
        print('Start Inference')
        image = np.reshape(image, [1, 160, 160, 160, 1])
        prediction1, prediction2 = self.prediction_p(image)

        I1 = np.reshape(np.squeeze(prediction1[0, :, :, :]), [160, 160, 1])
        I2 = np.reshape(np.squeeze(prediction2[0, :, :, :]), [160, 160, 1])
        I1 = np.tile(I1, [1, 1, 160])
        I2 = np.transpose(np.tile(I2, [1, 1, 160]), [2, 1, 0])

        return (I1 + I2) / 2


