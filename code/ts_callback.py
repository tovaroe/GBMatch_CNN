# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:33:56 2020

@author: Tom
"""

import tensorflow as tf
from tensorflow import keras as k
import numpy as np

class TSCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_flow_val,
                 validation_rounds = 1):
        super().__init__()
        self.batches = []
        
        print(f"Loading {validation_rounds} validation batches")
        for i in range(validation_rounds):
            print(i)
            batch = next(image_flow_val)
            self.batches.append(batch)
            
    def on_train_begin(self, logs=None):
        mse = k.metrics.MeanSquaredError()
        acc = k.metrics.Accuracy()
        
        for batch in self.batches:
            y_pred = self.model.predict(batch)
            y_true = batch[1]
            mse.update_state(y_true, y_pred)
            
            class_pred = np.argmax(y_pred, axis=1)
            class_true = np.argmax(y_true, axis=1)
            acc.update_state(class_pred, class_true)
            
        mse_result = mse.result()
        acc_result = acc.result()
            
        print(f' - val_mse at train begin: {mse_result:.4f}')
        print(f' - val_accuracy at train begin: {acc_result:.4f}')
            
    
    def on_epoch_end(self, epoch, logs=None):
        mse = k.metrics.MeanSquaredError()
        acc = k.metrics.Accuracy()
        
        for batch in self.batches:
            y_pred = self.model.predict(batch)
            y_true = batch[1]
            mse.update_state(y_true, y_pred)
            
            class_pred = np.argmax(y_pred, axis=1)
            class_true = np.argmax(y_true, axis=1)
            acc.update_state(class_pred, class_true)
            
        mse_result = mse.result()
        acc_result = acc.result()
            
        print(f' - val_mse: {mse_result:.4f}')
        print(f' - val_accuracy: {acc_result:.4f}')
        
        logs['val_loss'] = mse_result
        logs['val_accuracy'] = acc_result
    
        
        
        
        
                