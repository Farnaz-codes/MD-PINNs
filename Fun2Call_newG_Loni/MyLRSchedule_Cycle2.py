# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:54:36 2024

@author: fnazr
"""
import tensorflow as tf
from tensorflow import keras as keras
DTYPE='float32'

class MyLRSchedule_Cycle(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, LR_method = 'cycle'):
        self.LR_method = LR_method

    def __call__(self, step, max_lr = 9e-4 , base_lr = 1e-4):
        if  max_lr == 1: lr = 3e-6#6e-6  # last 6% of training time
        elif  max_lr == 0.5: lr = 4e-6#8e-6  # last 6% of training time
        elif  max_lr == 0.25: lr = 6e-6#1e-5  # last 6% of training time
        elif  max_lr == 0.75: lr = 2e-6#4e-6  # last 6% of training time
        elif self.LR_method=='cycle':
           if max_lr == 0: lr = base_lr  # first 6% of training time
           else:
               step_size = 1000
               cycle = tf.floor(1+step/(2*step_size))
               x = tf.abs(step/step_size - 2*cycle + 1)
               func = 0.95 **(step/step_size)
               lr = (base_lr + (max_lr-base_lr)*tf.reduce_max([0, (1-x)]))*func
        else:
            lr = max_lr
            
        return  tf.cast(lr, tf.float32)
#%%
   
#LR = MyLRSchedule_Cycle(1000) 
#step0 = tf.Variable(0, trainable=False) 
#optimizer_method = tf.keras.optimizers.Adam(learning_rate=LR(step0)) 
#ll = optimizer_method.learning_rate
#ll2 =LR(step0.assign(400))
    
    