#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:34:59 2020

@author: yunghx_macpro
"""

import tensorflow as TF

ImageSize = 160
MaskSize = 160
NUM_OF_CLASSESS = 2
import os

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'

channel = 12


def weight_variable(shape, name):
    initial = TF.truncated_normal(shape, stddev=0.01)
    return TF.get_variable(name=name, initializer=initial)


def bias_variable(shape, name):
    initial = TF.constant(0.01, shape=shape)
    return TF.get_variable(name=name, initializer=initial)


def conv3d(x, W):
    return TF.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def conv2d(x, W):
    return TF.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_de(x, name, factor=2):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        shape = x.get_shape()
        W = weight_variable([2, 2, shape[3].value, shape[3].value], name="W")
        b = bias_variable([shape[3].value], name="b_t3")
        Outshape = [shape[0].value, factor * shape[1].value, factor * shape[2].value, shape[3].value]
        return TF.nn.conv2d_transpose(x, W, output_shape=Outshape, strides=[1, factor, factor, 1], padding='SAME') + b


def max_pool3D(x):
    return TF.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def ReLu(x):
    return TF.nn.relu(x)


def Conv3D(x, kernel_size, input_ch, output_ch, name):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        W_C = weight_variable([kernel_size, kernel_size, kernel_size, input_ch, output_ch], 'W_C')
        L = conv3d(x, W_C)
        return L


def ResBolck2D(x, input_ch, output_ch, name, Phase_train, kernel_size=3):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        x1 = batch_norm(ReLu(Conv2D(x, kernel_size, input_ch, output_ch, 'C3D1')), Phase_train, 'IN1')
        x2 = batch_norm(ReLu(Conv2D(x1, kernel_size, input_ch, output_ch, 'C3D1')), Phase_train, 'IN1')
        x3 = batch_norm(ReLu(Conv2D(x2, kernel_size, input_ch, output_ch, 'C3D2')), Phase_train, 'IN2')
        return x + x3


def Bolck3D(x, input_ch, output_ch, name, Phase_train, kernel_size=3):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        x1 = batch_norm(ReLu(Conv3D(x, kernel_size, input_ch, output_ch, 'C3D1')), Phase_train, 'IN1')
        return x1


def Bolck2D(x, input_ch, output_ch, name, Phase_train, kernel_size=3):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        x1 = batch_norm(ReLu(Conv2D(x, kernel_size, input_ch, output_ch, 'C3D1')), Phase_train, 'IN1')
        return x1


def Conv2D(x, kernel_size, input_ch, output_ch, name):
    with TF.variable_scope(name, reuse=TF.AUTO_REUSE):
        W_C = weight_variable([kernel_size, kernel_size, input_ch, output_ch], 'W_C')
        L = (conv2d(x, W_C))
        return L


def batch_norm(x, phase_train, Name):
    return TF.contrib.layers.instance_norm(inputs=x,
                                           reuse=TF.AUTO_REUSE, scope=Name)


def DimensionDownConv(x, name, Phase_train):
    # Tensorflow standard [n, D, H, W, C]
    # Pytorch standard [n, C, D, H, W]
    with TF.variable_scope(name):
        Shape = x.get_shape()
        W_C1 = weight_variable([3, 3, 3, Shape[4].value, 1], 'W_C1')
        L1 = batch_norm(ReLu(conv3d(x, W_C1)), Phase_train, 'IN1')
        L1S = TF.reshape(L1, shape=[Shape[0].value, Shape[1].value, Shape[2].value, Shape[3].value])
        W_C2 = weight_variable([3, 3, Shape[3].value, 1], 'W_C2')
        L2 = batch_norm(ReLu(conv2d(L1S, W_C2)), Phase_train, 'IN2')
        W_C3 = weight_variable([3, 3, 1, Shape[4].value], 'W_C3')
        L3 = batch_norm(ReLu(conv2d(L2, W_C3)), Phase_train, 'IN3')
        return L3


def DimensionDown(x, name, Phase_train):
    with TF.variable_scope(name):
        Shape = x.get_shape()
        x1max = TF.reduce_max(x, reduction_indices=[3], keepdims=True)
        x2max = TF.reshape(x1max, [Shape[0].value, Shape[1].value, Shape[2].value, Shape[4].value])
        x1mean = TF.reduce_mean(x, reduction_indices=[3], keepdims=True)
        x2mean = TF.reshape(x1mean, [Shape[0].value, Shape[1].value, Shape[2].value, Shape[4].value])
        x_con = TF.concat([x2max, x2mean], axis=3)
        W_C1 = weight_variable([3, 3, 2 * Shape[4].value, Shape[4].value], 'W_C1')
        x_con_new = conv2d(x_con, W_C1)
        x_cov = DimensionDownConv(x, 'Convs', Phase_train)

        return x_con_new + x_cov


def Decoder_2D(x5, x4, x3, x2, keep_prob, Phase_train, activation_map_shapes):
    with TF.variable_scope('Decoder_2D', reuse=TF.AUTO_REUSE):
        x7 = ResBolck2D(x5, 8 * channel, 8 * channel, 'Res6', Phase_train)

        x8 = conv2d_de(x7, 'DC1') + x4
        x9c = ReLu(batch_norm(Conv2D(x8, 3, 8 * channel, 4 * channel, 'Conv8'), Phase_train, 'bn5'))
        x9 = Bolck2D(x9c, 4 * channel, 4 * channel, 'Res8', Phase_train)

        activation_map_shapes['up1'] = x9.get_shape()
        x10 = conv2d_de(x9, 'DC2') + x3
        x11c = ReLu(batch_norm(Conv2D(x10, 3, 4 * channel, 2 * channel, 'Conv11'), Phase_train, 'bn6'))
        x11 = Bolck2D(x11c, 2 * channel, 2 * channel, 'Res11', Phase_train)

        activation_map_shapes['up2'] = x11.get_shape()
        x12 = conv2d_de(x11, 'DC3') + x2
        x13c = ReLu(batch_norm(Conv2D(x12, 3, 2 * channel, 2 * channel, 'Conv13'), Phase_train, 'bn7'))
        x13 = Bolck2D(x13c, 2 * channel, 2 * channel, 'Res13', Phase_train)

        activation_map_shapes['up3'] = x13.get_shape()
        x14 = conv2d_de(x13, 'DC4')
        x15c = ReLu(batch_norm(Conv2D(x14, 3, 2 * channel, channel, 'Conv15'), Phase_train, 'bn8'))
        x15 = Bolck2D(x15c, channel, channel, 'Res15', Phase_train)

        activation_map_shapes['up4'] = x15.get_shape()
        Output = TF.nn.sigmoid(Conv2D(x15, 1, channel, 1, 'Output'))
        return Output, activation_map_shapes


def fcn_simple_net(image, keep_prob, phase_train, batch_size):
    activation_map_shapes = dict()
    with TF.variable_scope('Simple_Net', reuse=TF.AUTO_REUSE):
        activation_map_shapes['input'] = image.get_shape()

        x1c = ReLu(batch_norm(Conv3D(image, 7, 1, channel, 'Input'), phase_train, 'bn1'))
        x1 = Bolck3D(x1c, channel, channel, 'Res1', phase_train)
        x1p = (max_pool3D(x1))
        activation_map_shapes['down1'] = x1p.get_shape()

        x2c = ReLu(batch_norm(Conv3D(x1p, 3, channel, 2 * channel, 'Conv2'), phase_train, 'bn2'))
        x2 = Bolck3D(x2c, 2 * channel, 2 * channel, 'Res2', phase_train)
        x2p = (max_pool3D(x2))
        activation_map_shapes['down2'] = x2p.get_shape()

        x3c = ReLu(batch_norm(Conv3D(x2p, 3, 2 * channel, 4 * channel, 'Conv3'), phase_train, 'bn3'))
        x3 = Bolck3D(x3c, 4 * channel, 4 * channel, 'Res3', phase_train)
        x3p = (max_pool3D(x3))
        activation_map_shapes['down3'] = x3p.get_shape()

        x4c = ReLu(batch_norm(Conv3D(x3p, 3, 4 * channel, 8 * channel, 'Conv4'), phase_train, 'bn4'))
        x4 = Bolck3D(x4c, 8 * channel, 8 * channel, 'Res4', phase_train)
        x4p = (max_pool3D(x4))
        activation_map_shapes['down4'] = x4p.get_shape()

        x5 = Bolck3D(x4p, 8 * channel, 8 * channel, 'Res5', phase_train)
        activation_map_shapes['pre_reduction'] = x5.get_shape()

        x5_p = DimensionDown(TF.concat([x5, TF.transpose(x5, [0, 3, 2, 1, 4])], axis=0), 'D1', phase_train)
        activation_map_shapes['x5_d'] = x5_p.get_shape()
        x4_p = DimensionDown(TF.concat([x4, TF.transpose(x4, [0, 3, 2, 1, 4])], axis=0), 'D2', phase_train)
        activation_map_shapes['x4_d'] = x4_p.get_shape()
        x3_p = DimensionDown(TF.concat([x3, TF.transpose(x3, [0, 3, 2, 1, 4])], axis=0), 'D3', phase_train)
        activation_map_shapes['x3_d'] = x3_p.get_shape()
        x2_p = DimensionDown(TF.concat([x2, TF.transpose(x2, [0, 3, 2, 1, 4])], axis=0), 'D4', phase_train)
        activation_map_shapes['x2_d'] = x2_p.get_shape()

        Output, activation_map_shapes = Decoder_2D(x5_p, x4_p, x3_p, x2_p, keep_prob, phase_train, activation_map_shapes)
        Output_1 = Output[0:batch_size, :, :, :]
        Output_2 = Output[batch_size:2 * batch_size, :, :, :]

        activation_map_shapes = {k: v.as_list() for (k, v) in activation_map_shapes.items()}

        return Output_1, Output_2, activation_map_shapes

