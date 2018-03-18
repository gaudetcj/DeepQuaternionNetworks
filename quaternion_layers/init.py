#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chase Gaudet
# code based on work by Chiheb Trabelsi
# on Deep Complex Networks git source

import numpy as np
import scipy.stats as st
from random import gauss
from numpy.random import RandomState
from .dist import Chi4Random
import keras.backend as K
from keras import initializers
from keras.initializers import Initializer
from keras.utils.generic_utils import (serialize_keras_object,
                                       deserialize_keras_object)


class QuaternionInit(Initializer):
    # The standard quaternion initialization using
    # either the He or the Glorot criterion.
    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='he', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 31337 if seed is None else seed

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = initializers._compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / np.sqrt(2*(fan_in + fan_out))
        elif self.criterion == 'he':
            s = 1. / np.sqrt(2*fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = Chi4Random(s)
        flat_size = np.product(kernel_shape)
        modulus = rng.random(N=flat_size)
        modulus = modulus.reshape(kernel_shape)
        rng = RandomState(self.seed)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        
        # must make random unit vector for quaternion vector
        def make_rand_vector(dims):
            vec = [gauss(0, 1) for i in range(dims)]
            mag = sum(x**2 for x in vec) ** 0.5
            return [x/mag for x in vec]

        u_i = np.zeros(flat_size)
        u_j = np.zeros(flat_size)
        u_k = np.zeros(flat_size)
        for u in range(flat_size):
            unit = make_rand_vector(3)
            u_i[u] = unit[0]
            u_j[u] = unit[1]
            u_k[u] = unit[2]
        u_i = u_i.reshape(kernel_shape)
        u_j = u_j.reshape(kernel_shape)
        u_k = u_k.reshape(kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * u_i*np.sin(phase)
        weight_j = modulus * u_j*np.sin(phase)
        weight_k = modulus * u_k*np.sin(phase)
        weight = np.concatenate([weight_r, weight_i, weight_j, weight_k], axis=-1)

        return weight


class SqrtInit(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1 / K.sqrt(16), shape=shape, dtype=dtype)


# Aliases:
sqrt_init = SqrtInit
quaternion_init = QuaternionInit