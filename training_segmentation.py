#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Chase Gaudet
# code based on work by Chiheb Trabelsi
# on Deep Complex Networks git source

# Imports
import sys
sys.setrecursionlimit(10000)
import logging as L
import numpy as np
from complex_layers.utils import GetReal, GetImag
from complex_layers.conv import ComplexConv2D
from complex_layers.bn import ComplexBatchNormalization
from quaternion_layers.utils import Params, GetR, GetI, GetJ, GetK
from quaternion_layers.conv import QuaternionConv2D
from quaternion_layers.bn import QuaternionBatchNormalization
from batch_gen import gen_batch
import keras
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.datasets import cifar10, cifar100
from keras.layers import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import keras.backend as K
K.set_image_data_format('channels_first')
K.common.set_image_dim_ordering('th')


# Callbacks:
# Print a newline after each epoch.
class PrintNewlineAfterEpochCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		sys.stdout.write("\n")


# Keep a history of the validation performance.
class TrainValHistory(Callback):
	def __init__(self):
		self.train_loss = []
		self.val_loss   = []

	def on_epoch_end(self, epoch, logs={}):
		self.train_loss.append(logs.get('loss'))
		self.val_loss.append(logs.get('val_loss'))


def schedule(epoch):
    if epoch >= 0 and epoch < 10:
        lrate = 0.01
    elif epoch >= 10 and epoch < 50:
        lrate = 0.1
    elif epoch >= 50 and epoch < 100:
        lrate = 0.01
    elif epoch >= 100 and epoch < 150:
        lrate = 0.001
    elif epoch >= 150:
        lrate = 0.0001
    return lrate


def learnVectorBlock(I, featmaps, filter_size, act, bnArgs):
    """Learn initial vector component for input."""

    O = BatchNormalization(**bnArgs)(I)
    O = Activation(act)(O)
    O = Convolution2D(featmaps, filter_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    O = BatchNormalization(**bnArgs)(O)
    O = Activation(act)(O)
    O = Convolution2D(featmaps, filter_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    return O


def getResidualBlock(I, mode, filter_size, featmaps, activation, dropout, shortcut, convArgs, bnArgs):
    """Get residual block."""
    
    if mode == "real":
        O = BatchNormalization(**bnArgs)(I)
    elif mode == "complex":
        O = ComplexBatchNormalization(**bnArgs)(I)
    elif mode == "quaternion":
        O = QuaternionBatchNormalization(**bnArgs)(I)
    O = Activation(activation)(O)

    if shortcut == 'regular':
        if mode == "real":
            O = Conv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "complex":
            O = ComplexConv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "quaternion":
            O = QuaternionConv2D(featmaps, filter_size, **convArgs)(O)
    elif shortcut == 'projection':
        if mode == "real":
            O = Conv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "complex":
            O = ComplexConv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "quaternion":
            O = QuaternionConv2D(featmaps, filter_size, **convArgs)(O)

    if mode == "real":
        O = BatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = Conv2D(featmaps, filter_size, **convArgs)(O)
    elif mode == "complex":
        O = ComplexBatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = ComplexConv2D(featmaps, filter_size, **convArgs)(O)
    elif mode == "quaternion":
        O = QuaternionBatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = QuaternionConv2D(featmaps, filter_size, **convArgs)(O)

    if shortcut == 'regular':
        O = Add()([O, I])
    elif shortcut == 'projection':
        if mode == "real":
            X = Conv2D(featmaps, (1, 1), **convArgs)(I)
            O = Concatenate(1)([X, O])
        elif mode == "complex":
            X = ComplexConv2D(featmaps, (1, 1), **convArgs)(I)
            O_real = Concatenate(1)([GetReal()(X), GetReal()(O)])
            O_imag = Concatenate(1)([GetImag()(X), GetImag()(O)])
            O = Concatenate(1)([O_real, O_imag])
        elif mode == "quaternion":
            X = QuaternionConv2D(featmaps, (1, 1), **convArgs)(I)
            O_r = Concatenate(1)([GetR()(X), GetR()(O)])
            O_i = Concatenate(1)([GetI()(X), GetI()(O)])
            O_j = Concatenate(1)([GetJ()(X), GetJ()(O)])
            O_k = Concatenate(1)([GetK()(X), GetK()(O)])
            O = Concatenate(1)([O_r, O_i, O_j, O_k])

    return O

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def getModel(params):
    mode = params.mode
    n = params.num_blocks
    sf = params.start_filter
    activation = params.act
    dropout = params.dropout
    inputShape = (3, 93, 310)
    channelAxis = 1
    filsize = (3, 3)
    convArgs = {
    "padding": "same",
    "use_bias": False,
    "kernel_regularizer": l2(0.0001),
    }
    bnArgs = {
    "axis": channelAxis,
    "momentum": 0.9,
    "epsilon": 1e-04,
    "scale": False
    }

    convArgs.update({"kernel_initializer": params.init})

    # Create the vector channels
    R = Input(shape=inputShape)

    if mode != "quaternion":
        I = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        O = concatenate([R, I], axis=channelAxis)
    else:
        I = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        J = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        K = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        O = concatenate([R, I, J, K], axis=channelAxis)

    if mode == "real":
        O = Conv2D(sf, filsize, **convArgs)(O)
        O = BatchNormalization(**bnArgs)(O)
    elif mode == "complex":
        O = ComplexConv2D(sf, filsize, **convArgs)(O)
        O = ComplexBatchNormalization(**bnArgs)(O)
    else:
        O = QuaternionConv2D(sf, filsize, **convArgs)(O)
        O = QuaternionBatchNormalization(**bnArgs)(O)
    O = Activation(activation)(O)

    for i in range(n):
        O = getResidualBlock(O, mode, filsize, sf, activation, dropout, 'regular', convArgs, bnArgs)

    O = getResidualBlock(O, mode, filsize, sf, activation, dropout, 'projection', convArgs, bnArgs)

    for i in range(n-1):
        O = getResidualBlock(O, mode, filsize, sf*2, activation, dropout, 'regular', convArgs, bnArgs)

    O = getResidualBlock(O, mode, filsize, sf*2, activation, dropout, 'projection', convArgs, bnArgs)

    for i in range(n-1):
        O = getResidualBlock(O, mode, filsize, sf*4, activation, dropout, 'regular', convArgs, bnArgs)

    # heatmap output
    O = Convolution2D(1, 1, activation='sigmoid')(O)

    model = Model(R, O)
    opt = SGD (lr = params.lr,
               momentum = params.momentum,
               decay = params.decay,
               nesterov = True,
               clipnorm = params.clipnorm)
    model.compile(opt, dice_coef_loss)
    return model


def train(params, model):
    image_shape = (3, 93, 310)
    batch_size = params.batch_size
    epochs = params.num_epochs


    lrSchedCb = LearningRateScheduler(schedule)
    trainValHist = TrainValHistory()
    callbacks = [ModelCheckpoint('{}_weights.hd5'.format(params.mode), monitor='val_loss', verbose=0, save_best_only=True),
                 lrSchedCb,
                 trainValHist]

    t_gen = gen_batch(image_shape, 150)
    v_gen = gen_batch(image_shape, 50)
    for Xvb, Yvb in v_gen:
        Xv = Xvb
        Yv = Yvb 
        break

    e = 1
    while e <= epochs:
        Xt, Yt = next(t_gen)
        print('\nEPOCH: {}'.format(e))
        model.fit(Xt, Yt, 
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=(Xv,Yv))
        e += 1

    np.savetxt('{}_seg_train_loss.txt'.format(params.mode), trainValHist.train_loss)
    np.savetxt('{}_seg_val_loss.txt'.format(params.mode), trainValHist.val_loss)


if __name__ == '__main__':
    param_dict = {"mode": "quaternion",
                  "num_blocks": 3,
                  "start_filter": 8,
                  "dropout": 0,
                  "batch_size": 8,
                  "num_epochs": 200,
                  "act": "relu",
                  "init": "quaternion",
                  "lr": 1e-3,
                  "momentum": 0.9,
                  "decay": 0,
                  "clipnorm": 1.0
    }
    
    params = Params(param_dict)
    model = getModel(params)
    print(model.count_params())
    train(params, model)
