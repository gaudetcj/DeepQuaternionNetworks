#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chase Gaudet
# code based on work by Chiheb Trabelsi
# on Deep Complex Networks git source
#
# Implementation of Layer Normalization and Quaternion Layer Normalization


import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K
from .bn import QuaternionBN as quaternion_normalization
from .bn import sqrt_init 

def layernorm(x, axis, epsilon, gamma, beta):
    # assert self.built, 'Layer must be built before being called'
    input_shape = K.shape(x)
    reduction_axes = list(range(K.ndim(x)))
    del reduction_axes[axis]
    del reduction_axes[0]
    broadcast_shape = [1] * K.ndim(x)
    broadcast_shape[axis] = input_shape[axis]
    broadcast_shape[0] = K.shape(x)[0]

    # Perform normalization: centering and reduction

    mean = K.mean(x, axis=reduction_axes)
    broadcast_mean = K.reshape(mean, broadcast_shape)
    x_centred = x - broadcast_mean
    variance  = K.mean(x_centred ** 2, axis=reduction_axes) + epsilon
    broadcast_variance = K.reshape(variance, broadcast_shape)

    x_normed = x_centred / K.sqrt(broadcast_variance)

    # Perform scaling and shifting

    broadcast_shape_params = [1] * K.ndim(x)
    broadcast_shape_params[axis] = K.shape(x)[axis]
    broadcast_gamma  = K.reshape(gamma, broadcast_shape_params)
    broadcast_beta  = K.reshape(beta,  broadcast_shape_params)

    x_LN = broadcast_gamma * x_normed + broadcast_beta

    return x_LN

class LayerNormalization(Layer):
    
    def __init__(self,
                 epsilon=1e-4,
                 axis=-1,
                 beta_init='zeros',
                 gamma_init='ones',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: input_shape[self.axis]})
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))

        self.built = True

    def call(self, x, mask=None):
        assert self.built, 'Layer must be built before being called'
        return layernorm(x, self.axis, self.epsilon, self.gamma, self.beta)

    def get_config(self):
        config = {'epsilon':           self.epsilon,
                  'axis':              self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer':  self.beta_regularizer.get_config()  if self.beta_regularizer  else None
                  }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuaternionLayerNorm(Layer):
    def __init__(self,
                 epsilon=1e-4,
                 axis=-1,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer=sqrt_init,
                 gamma_off_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):

        self.supports_masking = True
        self.epsilon = epsilon
        self.axis = axis
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_diag_initializer = initializers.get(gamma_diag_initializer)
        self.gamma_off_initializer = initializers.get(gamma_off_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        super(QuaternionLayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):

        ndim = len(input_shape)
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        gamma_shape = (input_shape[self.axis] // 4,)
        if self.scale:
            self.gamma_rr = self.add_weight(
                shape=gamma_shape,
                name='gamma_rr',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            self.gamma_ii = self.add_weight(
                shape=gamma_shape,
                name='gamma_ii',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            self.gamma_jj = self.add_weight(
                shape=gamma_shape,
                name='gamma_jj',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            self.gamma_kk = self.add_weight(
                shape=gamma_shape,
                name='gamma_kk',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            self.gamma_ri = self.add_weight(
                shape=gamma_shape,
                name='gamma_ri',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
            self.gamma_rj = self.add_weight(
                shape=gamma_shape,
                name='gamma_rj',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
            self.gamma_rk = self.add_weight(
                shape=gamma_shape,
                name='gamma_rk',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
            self.gamma_ij = self.add_weight(
                shape=gamma_shape,
                name='gamma_ij',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
            self.gamma_ik = self.add_weight(
                shape=gamma_shape,
                name='gamma_ik',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
            self.gamma_jk = self.add_weight(
                shape=gamma_shape,
                name='gamma_jk',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_jj = None
            self.gamma_kk = None
            self.gamma_ri = None
            self.gamma_rj = None
            self.gamma_rk = None
            self.gamma_ij = None
            self.gamma_ik = None
            self.gamma_jk = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        ndim = K.ndim(inputs)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        del reduction_axes[0]
        input_dim = input_shape[self.axis] // 4
        mu = K.mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * ndim
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu_shape[0] = K.shape(inputs)[0]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_r = centred_squared[:, :input_dim]
            centred_squared_i = centred_squared[:, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, input_dim*3:]
            centred_r = input_centred[:, :input_dim]
            centred_i = input_centred[:, input_dim:input_dim*2]
            centred_j = input_centred[:, input_dim*2:input_dim*3]
            centred_k = input_centred[:, input_dim*3:]
        elif ndim == 3:
            centred_squared_r = centred_squared[:, :, :input_dim]
            centred_squared_i = centred_squared[:, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, input_dim*3:]
            centred_r = input_centred[:, :, :input_dim]
            centred_i = input_centred[:, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, input_dim*3:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_r = centred_squared[:, :, :, :input_dim]
            centred_squared_i = centred_squared[:, :, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, :, input_dim*3:]
            centred_r = input_centred[:, :, :, :input_dim]
            centred_i = input_centred[:, :, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, :, input_dim*3:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_r = centred_squared[:, :, :, :, :input_dim]
            centred_squared_i = centred_squared[:, :, :, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, :, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, :, :, input_dim*3:]
            centred_r = input_centred[:, :, :, :, :input_dim]
            centred_i = input_centred[:, :, :, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, :, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, :, :, input_dim*3:]
        else:
            raise ValueError(
                'Incorrect Layernorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_r,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_i,
                axis=reduction_axes
            ) + self.epsilon
            Vjj = K.mean(
                centred_squared_j,
                axis=reduction_axes
            ) + self.epsilon
            Vkk = K.mean(
                centred_squared_k,
                axis=reduction_axes
            ) + self.epsilon
            Vri = K.mean(
                centred_r * centred_i,
                axis=reduction_axes,
            )
            Vrj = K.mean(
                centred_r * centred_j,
                axis=reduction_axes,
            )
            Vrk = K.mean(
                centred_r * centred_k,
                axis=reduction_axes,
            )
            Vij = K.mean(
                centred_i * centred_j,
                axis=reduction_axes,
            )
            Vik = K.mean(
                centred_i * centred_k,
                axis=reduction_axes,
            )
            Vjk = K.mean(
                centred_j * centred_k,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vjj = None
            Vkk = None
            Vri = None
            Vrj = None
            Vrk = None
            Vij = None
            Vik = None
            Vkk = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        return quaternion_normalization(
            input_centred, 
            Vrr, Vri, Vrj, Vrk, Vii, 
            Vij, Vik, Vjj, Vjk, Vkk,
            self.beta, 
            self.gamma_rr, self.gamma_ri, 
            self.gamma_rj, self.gamma_rk, 
            self.gamma_ii, self.gamma_ij, 
            self.gamma_ik, self.gamma_jj, 
            self.gamma_jk, self.gamma_kk,
            self.scale, self.center,
            layernorm=True, axis=self.axis
        )

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer),
            'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(QuaternionLayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
