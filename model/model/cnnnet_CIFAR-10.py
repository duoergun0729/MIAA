from __future__ import print_function
import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
from keras.layers import Conv2D, MaxPool2D, Flatten,Reshape
import numpy as np
np.random.seed(10000)

#https://keras.io/examples/cifar10_cnn/


#MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples
def model_user(input_shape,labels_dim):
    
    
    inputs=Input(shape=input_shape)
    middle_layer=Reshape((32, 32,3))(inputs)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    #middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Conv2D(64, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(64, (3,3), activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    #middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Flatten()(middle_layer)
    middle_layer=Dense(512,activation='relu')(middle_layer)
    #middle_layer=Dropout(0.5)(middle_layer)
    outputs_logits=Dense(labels_dim)(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

#MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples
def model_user_dropout(input_shape,labels_dim):
    
    
    inputs=Input(shape=input_shape)
    middle_layer=Reshape((32, 32,3))(inputs)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Conv2D(64, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(64, (3,3), activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Flatten()(middle_layer)
    middle_layer=Dense(512,activation='relu')(middle_layer)
    middle_layer=Dropout(0.5)(middle_layer)
    outputs_logits=Dense(labels_dim)(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def model_user_l2(input_shape,labels_dim,e=0.005):
    
    inputs=Input(shape=input_shape)
    middle_layer=Reshape((32, 32,3))(inputs)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(32, (3,3),activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    #middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Conv2D(64, (3,3),activation='relu')(middle_layer)
    middle_layer=Conv2D(64, (3,3), activation='relu')(middle_layer)
    middle_layer=MaxPool2D(pool_size=(2,2))(middle_layer)
    #middle_layer=Dropout(0.25)(middle_layer)
    
    middle_layer=Flatten()(middle_layer)
    middle_layer=Dense(512,activation='relu')(middle_layer)
    #middle_layer=Dropout(0.5)(middle_layer)
    outputs_logits=Dense(labels_dim, 
                         kernel_regularizer=l2(e) )(middle_layer)
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
    """
    inputs_b=Input(shape=input_shape)
    x_b=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    """
    inputs_b=Input(shape=input_shape)
    x_b=Dense(16,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(8,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    
    return model   