# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:33:56 2020

@author: Tom
"""

import tensorflow as tf
import numpy as np
from sksurv.metrics import concordance_index_censored as cindex
import pandas as pd

class CIndexCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_flow_val, idx, followup, event,
                 loss_func = None,
                 validation_rounds = 1):
        super().__init__()
        self.image_flow_val = image_flow_val
        self.df_surv = pd.DataFrame(dict(idx=idx, followup=followup, event=event, y_pred=np.nan)).set_index('idx')
        self.loss_func = loss_func
        self.validation_rounds = validation_rounds
        
    def on_train_begin(self, logs=None):
        losses = []
        c_indices = []
        
        for j in range(self.validation_rounds):
            for i in range(len(self.image_flow_val)):
                batch = next(self.image_flow_val)
                ids = batch[1].tolist() # ids +
                y_pred = np.squeeze(self.model.predict(batch)).tolist() # y_pred +
                self.df_surv.loc[ids, 'y_pred'] = y_pred
                
                if self.loss_func:
                    loss = self.loss_func(tf.constant(ids, dtype='int64'), tf.constant(y_pred, dtype='float32')).numpy().mean()
                    losses.append(np.float64(loss))
    
            c_indices.append(np.float64(cindex(self.df_surv.event.astype(bool), self.df_surv.followup, self.df_surv.y_pred)[0]))
            
            self.df_surv.y_pred = np.nan
        
        if self.loss_func:
            mean_loss = np.round(np.mean(losses), 3)
            print(f'- val_loss at train begin: {mean_loss}')        

        
        mean_c_index = np.round(np.mean(c_indices), 3)
        print(f'- val_cindex at train begin: {mean_c_index}')        
    
    def on_epoch_end(self, epoch, logs=None):
        losses = []
        c_indices = []
        
        for j in range(self.validation_rounds):
            for i in range(len(self.image_flow_val)):
                batch = next(self.image_flow_val)
                ids = batch[1].tolist() # ids +
                y_pred = np.squeeze(self.model.predict(batch)).tolist() # y_pred +
                self.df_surv.loc[ids, 'y_pred'] = y_pred
                
                if self.loss_func:
                    loss = self.loss_func(tf.constant(ids, dtype='int64'), tf.constant(y_pred, dtype='float32')).numpy().mean()
                    losses.append(np.float64(loss))
    

            c_indices.append(np.float64(cindex(self.df_surv.event.astype(bool), self.df_surv.followup, self.df_surv.y_pred)[0]))
            
            self.df_surv.y_pred = np.nan
        
        if self.loss_func:
            mean_loss = np.round(np.mean(losses), 3)
            print(f'- val_loss: {mean_loss}')        
            logs['val_loss'] = mean_loss
        
        mean_c_index = np.round(np.mean(c_indices), 3)
        print(f'- val_cindex: {mean_c_index}')
        logs['val_cindex'] = mean_c_index
        
        
                