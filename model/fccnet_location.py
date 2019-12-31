from __future__ import print_function
import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
np.random.seed(10000)

import logging
import logging.config

logging.config.fileConfig('log.ini')
logger = logging.getLogger('api')


def model_user(input_shape,labels_dim):
    inputs=Input(shape=input_shape)
    middle_layer=Dense(1024,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_user_l2(input_shape,labels_dim,e=0.005):
    
    inputs=Input(shape=input_shape)
    middle_layer=Dense(1024,kernel_initializer=keras.initializers.glorot_uniform(seed=100),
                       kernel_regularizer=l2(e),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),
                       kernel_regularizer=l2(e),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),
                       kernel_regularizer=l2(e),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),
                       kernel_regularizer=l2(e),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100),
                        kernel_regularizer=l2(e))(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_user_dropout(input_shape,labels_dim,e=0.1):
    inputs=Input(shape=input_shape)
    middle_layer=Dense(1024,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    
    middle_layer=Dropout(e)(middle_layer)
    
    outputs_logits=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_defense_optimize(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_attack_nn(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model  

def model_attack_nn_H(input_shape,labels_dim,H):
    inputs_b=Input(shape=input_shape)
    
    i=H-1
    
    k=512
    x_b=Dense(k,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    k/=2
    for i in range(1,H):
        x_b=Dense(k,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
        k/=2
    
    
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model   