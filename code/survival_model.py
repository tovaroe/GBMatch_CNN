# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:06:38 2020

@author: Tom
"""

import tensorflow as tf
from tensorflow import keras as k
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

class SurvivalModel(object):
    def __init__(self,
                 model,
                 base_model,
                 base_model_name,
                 input_shape,
                 dropout):
        
        self.model = model
        self.base_model = base_model
        self.history = []
        self.trainable_from_layer = None
        
        self._BASE_MODEL_NAME = base_model_name
        self._INPUT_SHAPE = input_shape
        self._DROPOUT = dropout
        
    @classmethod
    def build_new(cls,
                  base_model_name='Xception', 
                  input_shape=(512, 512, 3),
                  dropout=0.25,
                  target='OS'):
        

        
        inputs = k.layers.Input(input_shape, dtype=tf.uint8)
        x = tf.cast(inputs, tf.float32)
            
        if base_model_name=='Xception':
            base_model = k.applications.Xception(
                weights='imagenet',
                input_shape=input_shape,
                include_top=False)          
            x = k.applications.xception.preprocess_input(x)
        elif base_model_name=='ResNet50':
            base_model = k.applications.ResNet50V2(
                weights='imagenet',
                input_shape=input_shape,
                include_top=False)
            x = k.applications.resnet_v2.preprocess_input(x)       
        else:
            raise ValueError("%s is not a valid base model" % (base_model_name))
        
            
        for layer in base_model.layers: layer.trainable = False # needs to be done on each layer, so that setting specific layers trainable works smoothly            
        x = base_model(x, training=False)               
        x = k.layers.GlobalAveragePooling2D()(x)               
        kernel_regularizer=None
        x = k.layers.Dropout(dropout)(x)
        
        if target=='OS':
            outputs = k.layers.Dense(1, activation='linear', kernel_regularizer=kernel_regularizer)(x)
        elif target=='TS':
            outputs = k.layers.Dense(3, activation='softmax', kernel_regularizer=kernel_regularizer)(x)
        else:
            raise ValueError(f"Target must be either 'OS' or 'TS', got {target} instead.")
        model = k.Model(inputs, outputs)
                   
        return cls(model, base_model, base_model_name, input_shape, dropout)

    
    @classmethod
    def load(cls, path, target):
        
        with open(path + 'model_attributes.pickle', 'rb') as handle:
            model_attributes = pickle.load(handle)
            history = model_attributes['history']
            base_model_name = model_attributes['base_model_name']
            input_shape = model_attributes['input_shape']
            dropout = model_attributes['dropout']
            trainable_from_layer = model_attributes['trainable_from_layer']
            
        s_model = cls.build_new(base_model_name = base_model_name,
                                input_shape = input_shape,
                                dropout = dropout,
                                target=target)
        s_model.history = history
        
        if not trainable_from_layer==None: 
            s_model.set_base_trainable(from_layer=trainable_from_layer, to_layer=None)
            
        s_model.model.load_weights(path + 'model_weights.h5')
                
        return s_model           
    
    def save(self, path):
        # model.save has loading bugs in tf 2.1; works in tf 2.2 (not available via anaconda at the time of coding this part), so I'm using a workaround here
        
        if not os.path.exists(path): os.mkdir(path)
        self.model.save_weights(path + 'model_weights.h5')
        
        model_attributes = {'history': self.history,
                            'base_model_name': self._BASE_MODEL_NAME,
                            'input_shape': self._INPUT_SHAPE,
                            'dropout': self._DROPOUT,
                            'trainable_from_layer': self.trainable_from_layer}
        
        with open(path+'model_attributes.pickle', 'wb') as handle:
            pickle.dump(model_attributes, handle)
        
        self.plot_history(path)
        
        # Saving the optimizer state - might be useful... https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = k.backend.batch_get_value(symbolic_weights)
        with open(path+'optimizer.pickle', 'wb') as handle:
            pickle.dump(weight_values, handle)
    
    def compile(self,
                optimizer = k.optimizers.Adam(0.001),
                loss = k.losses.BinaryCrossentropy(from_logits = True),
                metrics = None,
                **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        
    def fit(self,
            data_handler,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        
        if 'steps_per_epoch' == None:
            batch_size = data_handler._BATCH_SIZE
            n = data_handler.train_generator.n
            steps_per_epoch = np.ceil(n/batch_size)
        if  'validation_steps' == None:
            batch_size = data_handler._BATCH_SIZE
            n = data_handler.val_generator.n
            validation_steps = np.ceil(n/batch_size)
        

        
        history = self.model.fit(data_handler.train_generator, 
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 **kwargs)
        self.history.append(history.history)       
            
    def set_base_trainable(self, from_layer=None, to_layer=None, trainable=True):
        # defaults to setting all layers trainable
        self.trainable_from_layer = from_layer
        
        for layer in self.base_model.layers[from_layer:to_layer]:
            layer.trainable = trainable
        
    def summary(self):
        self.model.summary()   
        
    def get_trained_epochs(self):
        d = {}
        for h in self.history:
            for key in h.keys():
                if key not in d:
                    d[key] = h[key].copy()
                else:
                    d[key] += h[key]
        if not 'loss' in d.keys():
            return 0
        return len(d['loss'])
        
    def plot_history(self, save_path=None):
        
        d = {}
        for h in self.history:
            for key in h.keys():
                if key not in d:
                    d[key] = h[key].copy()
                else:
                    d[key] += h[key]
                         
        keys = [key for key in d if not key.startswith('val_')]
       
        for key in keys:
            plt.figure()
            values = d[key]
            epochs = range(1, len(values)+1)
            plt.plot(epochs, values, 'r', label='training')
            if ('val_' + key) in d:
                plt.plot(epochs, d['val_' + key], 'b', label='validation')
            plt.title(key)
            plt.legend()
            if save_path:
                plt.savefig(save_path + key + '.png')
            plt.show()
            
        if 'val_cindex' in d:
            key = 'val_cindex'
            plt.figure()
            values = d[key]
            epochs = range(1, len(values)+1)
            plt.plot(epochs, values, 'r', label='validation')
            plt.title(key)
            plt.legend()
            if save_path:
                plt.savefig(save_path + key + '.png')
            plt.show()
            
        if 'val_accuracy' in d:
            key = 'val_accuracy'
            plt.figure()
            values = d[key]
            epochs = range(1, len(values)+1)
            plt.plot(epochs, values, 'r', label='validation')
            plt.title(key)
            plt.legend()
            if save_path:
                plt.savefig(save_path + key + '.png')
            plt.show()