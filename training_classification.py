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

# Also evaluate performance on test set at each epoch end.
class TestErrorCallback(Callback):
	def __init__(self, test_data):
		self.test_data    = test_data
		self.loss_history = []
		self.acc_history  = []

	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data
		
		L.getLogger("train").info("Epoch {:5d} Evaluating on test set...".format(epoch+1))
		test_loss, test_acc = self.model.evaluate(x, y, verbose=0)
		L.getLogger("train").info("                                      complete.")
		
		self.loss_history.append(test_loss)
		self.acc_history.append(test_acc)
		
		L.getLogger("train").info("Epoch {:5d} train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}, test_loss: {}, test_acc: {}".format(
		                          epoch+1,
		                          logs["loss"],     logs["acc"],
		                          logs["val_loss"], logs["val_acc"],
		                          test_loss,        test_acc))

# Keep a history of the validation performance.
class TrainValHistory(Callback):
	def __init__(self):
		self.train_loss = []
		self.train_acc  = []
		self.val_loss   = []
		self.val_acc    = []

	def on_epoch_end(self, epoch, logs={}):
		self.train_loss.append(logs.get('loss'))
		self.train_acc .append(logs.get('acc'))
		self.val_loss  .append(logs.get('val_loss'))
		self.val_acc   .append(logs.get('val_acc'))


class LrDivisor(Callback):
    def __init__(self, patience=float(50000), division_cst=10.0, epsilon=1e-03, verbose=1, epoch_checkpoints={41, 61}):
        super(Callback, self).__init__()
        self.patience = patience
        self.checkpoints = epoch_checkpoints
        self.wait = 0
        self.previous_score = 0.
        self.division_cst = division_cst
        self.epsilon = epsilon
        self.verbose = verbose
        self.iterations = 0

    def on_batch_begin(self, batch, logs={}):
        self.iterations += 1

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        divide = False
        if (epoch + 1) in self.checkpoints:
            divide = True
        elif (current_score >= self.previous_score - self.epsilon and current_score <= self.previous_score + self.epsilon):
            self.wait +=1
            if self.wait == self.patience:
                divide = True
        else:
            self.wait = 0
        if divide == True:
            K.set_value(self.model.optimizer.lr, self.model.optimizer.lr.get_value() / self.division_cst)
            self.wait = 0
            if self.verbose > 0:
                L.getLogger("train").info("Current learning rate is divided by"+str(self.division_cst) + ' and his values is equal to: ' + str(self.model.optimizer.lr.get_value()))
        self.previous_score = current_score


def schedule(epoch):
    if   epoch >=   0 and epoch <  10:
        lrate = 0.01
        if epoch == 0:
            L.getLogger("train").info("Current learning rate value is "+str(lrate))
    elif epoch >=  10 and epoch < 100:
        lrate = 0.01
        if epoch == 10:
            L.getLogger("train").info("Current learning rate value is "+str(lrate))
    elif epoch >= 100 and epoch < 120:
        lrate = 0.01
        if epoch == 100:
            L.getLogger("train").info("Current learning rate value is "+str(lrate))
    elif epoch >= 120 and epoch < 150:
        lrate = 0.001
        if epoch == 120:
            L.getLogger("train").info("Current learning rate value is "+str(lrate))
    elif epoch >= 150:
        lrate = 0.0001
        if epoch == 150:
            L.getLogger("train").info("Current learning rate value is "+str(lrate))
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


def getResidualBlock(I, mode, filter_size, featmaps, activation, shortcut, convArgs, bnArgs):
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
            O = Conv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)
        elif mode == "complex":
            O = ComplexConv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)
        elif mode == "quaternion":
            O = QuaternionConv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)

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
            X = Conv2D(featmaps, (1, 1), strides = (2, 2), **convArgs)(I)
            O = Concatenate(1)([X, O])
        elif mode == "complex":
            X = ComplexConv2D(featmaps, (1, 1), strides = (2, 2), **convArgs)(I)
            O_real = Concatenate(1)([GetReal()(X), GetReal()(O)])
            O_imag = Concatenate(1)([GetImag()(X), GetImag()(O)])
            O = Concatenate(1)([O_real, O_imag])
        elif mode == "quaternion":
            X = QuaternionConv2D(featmaps, (1, 1), strides = (2, 2), **convArgs)(I)
            O_r = Concatenate(1)([GetR()(X), GetR()(O)])
            O_i = Concatenate(1)([GetI()(X), GetI()(O)])
            O_j = Concatenate(1)([GetJ()(X), GetJ()(O)])
            O_k = Concatenate(1)([GetK()(X), GetK()(O)])
            O = Concatenate(1)([O_r, O_i, O_j, O_k])

    return O


def getModel(params):
    mode = params.mode
    n = params.num_blocks
    sf = params.start_filter
    dataset = params.dataset
    activation = params.act
    inputShape = (3, 32, 32)
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
    "epsilon": 1e-04
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
        O = getResidualBlock(O, mode, filsize, sf, activation, 'regular', convArgs, bnArgs)

    O = getResidualBlock(O, mode, filsize, sf, activation, 'projection', convArgs, bnArgs)

    for i in range(n-1):
        O = getResidualBlock(O, mode, filsize, sf*2, activation, 'regular', convArgs, bnArgs)

    O = getResidualBlock(O, mode, filsize, sf*2, activation, 'projection', convArgs, bnArgs)

    for i in range(n-1):
        O = getResidualBlock(O, mode, filsize, sf*4, activation, 'regular', convArgs, bnArgs)

    O = AveragePooling2D(pool_size=(8, 8))(O)

    # Flatten
    O = Flatten()(O)

    # Dense
    if dataset == 'cifar10':
        O = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(O)
    elif dataset == 'cifar100':
        O = Dense(100, activation='softmax', kernel_regularizer=l2(0.0001))(O)

    model = Model(R, O)
    opt = SGD (lr = params.lr,
               momentum = params.momentum,
               decay = params.decay,
               nesterov = True,
               clipnorm = params.clipnorm)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    return model


def train(params, model):
    if params.dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        nb_classes = 10
        n_train = 45000
    elif params.dataset == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        nb_classes = 100
        n_train = 45000

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    shuf_inds = np.arange(len(y_train))
    np.random.seed(424242)
    np.random.shuffle(shuf_inds)
    train_inds = shuf_inds[:n_train]
    val_inds = shuf_inds[n_train:]

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train_split = X_train[train_inds]
    X_val_split = X_train[val_inds]
    y_train_split = y_train[train_inds]
    y_val_split = y_train[val_inds]

    pixel_mean = np.mean(X_train_split, axis=0)

    X_train = X_train_split.astype(np.float32) - pixel_mean
    X_val = X_val_split.astype(np.float32) - pixel_mean
    X_test = X_test.astype(np.float32) - pixel_mean

    Y_train = to_categorical(y_train_split, nb_classes)
    Y_val = to_categorical(y_val_split, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(height_shift_range=0.125,
                                 width_shift_range=0.125,
                                 horizontal_flip=True)

    testErrCb = TestErrorCallback((X_test, Y_test))
    trainValHistCb = TrainValHistory()
    lrSchedCb = LearningRateScheduler(schedule)
    callbacks = [ModelCheckpoint('{}_weights.hd5'.format(params.mode), monitor='val_loss', verbose=0, save_best_only=True),
                 testErrCb,
                 lrSchedCb,
                 trainValHistCb]

    model.fit_generator(generator=datagen.flow(X_train, Y_train, batch_size=params.batch_size),
                        steps_per_epoch=(len(X_train)+params.batch_size-1) // params.batch_size,
                        epochs=params.num_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_val, Y_val))

    # Dump histories.
    np.savetxt('{}_test_loss.txt'.format(params.mode), np.asarray(testErrCb.loss_history))
    np.savetxt('{}_test_acc.txt'.format(params.mode), np.asarray(testErrCb.acc_history))
    np.savetxt('{}_train_loss.txt'.format(params.mode), np.asarray(trainValHistCb.train_loss))
    np.savetxt('{}_train_acc.txt'.format(params.mode), np.asarray(trainValHistCb.train_acc))
    np.savetxt('{}_val_loss.txt'.format(params.mode), np.asarray(trainValHistCb.val_loss))
    np.savetxt('{}_val_acc.txt'.format(params.mode), np.asarray(trainValHistCb.val_acc))


if __name__ == '__main__':
    param_dict = {"mode": "quaternion",
                  "num_blocks": 10,
                  "start_filter": 24,
                  "dropout": 0,
                  "batch_size": 32,
                  "num_epochs": 200,
                  "dataset": "cifar100",
                  "act": "relu",
                  "init": "quaternion",
                  "lr": 1e-3,
                  "momentum": 0.9,
                  "decay": 0,
                  "clipnorm": 1.0
    }
    
    params = Params(param_dict)
    model = getModel(params)
    train(params, model)
