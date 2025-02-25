# -*- coding: utf-8 -*-
"""
Model_network
Created on Fri Apr  7 15:36:49 2023

@author: fnazr
"""
import tensorflow as tf
#import numpy as np
DTYPE='float32'
P_range = [0.5,1.0]
tf.random.set_seed(196)
#silu
#%%---------------------------------------------------------------------------%

@tf.keras.saving.register_keras_serializable()
class Scaledtanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Scaledtanh, self).__init__(**kwargs)
        self.A = tf.Variable(initial_value=1/2, trainable=True, dtype=tf.float32)#, name='A')
        self.S = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)#, name='S')

    
    def call(self, inputs):
        n = 2 
        #return self.A * tf.keras.activations.swish(inputs)
        return self.S*tf.math.tanh(n*self.A * inputs)
    
    def get_config(self):
        config = super(Scaledtanh, self).get_config()
        return config
#%%---------------------------------------------------------------------------%
# Define the PINN model
#@tf.keras.saving.register_keras_serializable()
class PINN_model(tf.keras.Model):
    def __init__(self, act_fun = 'tanh', layers_ = [], LBUB = [] ,  dropout_rate = 0): 
        # dropout_rate = 0.1
        # self.act_fun = 'tanh' | 'swish' | Adapt_tanh
        super(PINN_model, self).__init__()
        self.layers_ = layers_
        self.dropout_rate = dropout_rate
        self.LBUB = LBUB
        self.act_fun = act_fun
           
        if isinstance(self.LBUB, list) and len(self.LBUB) == 0:
            self.lb_ , self.ub_ =[] ,[]
        else:
            self.lb_ = tf.convert_to_tensor(self.LBUB[0],dtype=DTYPE)
            self.ub_ = tf.convert_to_tensor(self.LBUB[1],dtype=DTYPE)
        #--------------------------------------------------------------------#
        tf.print(f'***********************{self.act_fun}*********************')
        self.model = tf.keras.Sequential()#,layers = layers)

        W_init = 'glorot_normal'  # 'he_uniform' or 'he_normal'
        B_init = 'zeros'
        #Act =  #sliu #'tanh' #ScaledSwish(),
        for i in range(len(layers_)-2):
            if self.act_fun== 'Adapt_tanh': Act = Scaledtanh()
            else: Act = self.act_fun
            #tf.keras.layers.Dense is already implemented as a TensorFlow graph operation.
            self.model.add(tf.keras.layers.Dense( units = layers_[i+1], 
                                             activation = Act, 
                                     kernel_initializer = W_init,
                                       bias_initializer = B_init,
                                                  dtype = DTYPE))  #kernel_regularizer=self.regularizer
            # add drop out
            if self.dropout_rate > 0:
                self.model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        # 'softplus'
        act2 = 'linear' if self.act_fun== 'swish'else 'softplus'
        self.model.add(tf.keras.layers.Dense(units = layers_[-1], 
                                        activation = act2, 
                                kernel_initializer = W_init,
                                  bias_initializer = B_init,
                                             dtype = DTYPE))# kernel_regularizer=self.regularizer,
    #-------------------------------------------------------------------------#
    def call(self, inputs =[]):
        # returns the output of the model with input X
        # if training:
        # H = 2.0*(X - self.lb[key_])/(self.ub[key_] - self.lb[key_]) - 1.0
        #  x = self.dropout(x, training=training)
        if isinstance(inputs, list) and len(inputs) == 0:
            inputs = tf.keras.Input(shape=(self.layers_[0],),dtype = DTYPE)
        inputs = tf.cast(inputs,dtype = DTYPE)
        if tf.is_tensor(self.lb_):# and self.lb_.shape[1]==4:
            Lb = tf.concat((self.lb_,tf.constant([[P_range[0]]],dtype = DTYPE)),axis=1)
            Ub = tf.concat((self.ub_,tf.constant([[P_range[1]]],dtype = DTYPE)),axis=1)
            inputs = 2.0*(inputs - Lb)/(Ub - Lb) - 1.0
        else:
            inputs = 2.0*(inputs - self.lb_)/(self.ub_ - self.lb_) - 1.0
            
        u = self.model(inputs)
        return u
    #-------------------------------------------------------------------------#
    def get_config(self):
        config = super(PINN_model, self).get_config()
        config.update({
            'act_fun': self.act_fun,'layers_': self.layers_, 'LBUB':self.LBUB})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
#%%----------------------------------------------------------------------------#
#input_ = tf.keras.Input(shape = (5,),dtype = DTYPE)
#Model = PINN_model(act_fun = 'Adapt_tanh', layers_ = [5,10,10,10,1],LBUB=([[0,0,0,0]],[[1,1,1,1]]))
#output_ = Model(input_)

