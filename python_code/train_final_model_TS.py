# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:00:53 2022

@author: Tom
"""

import tensorflow as tf
from tensorflow import keras as k
tf.config.list_physical_devices()
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
import shutil
import numpy as np
from scipy.stats import mode

import config
from data_handler import DataHandler
from survival_model import SurvivalModel



def main():

    model_save_path = config.model_save_path[:-1] + '_final/'

    dh = DataHandler(config.image_path, config.annotation_path, magnification=config.MAGNIFICATION,
                     y_col=config.Y_COL,
                     class_mode=config.CLASS_MODE,
                     im_format=config.IM_FORMAT,
                     im_size=config.IM_SIZE, 
                     rescale=config.RESCALE,
                     oversample_LTS=False,
                     kfold_val=None)
    

    
    loss = 'mse'
    
    
    print("Building new model...")
    survmodel = SurvivalModel.build_new(input_shape=(config.IM_SIZE, config.IM_SIZE, 3),
                                        base_model_name=config.BASE_MODEL,
                                        dropout=config.DROPOUT,
                                        target=config.target)
    survmodel.compile(optimizer=config.OPTIMIZER_1,
                      loss = loss)
    survmodel.fit(dh, epochs=config.EPOCHS_1, 
                  steps_per_epoch=config.STEPS_PER_EPOCH,
                  workers=config.WORKERS,
                  max_queue_size=config.MAX_QUEUE_SIZE)
    
    survmodel.save(model_save_path)
    
    
                
    for layer_index in range(config.LAYER_FINETUNING_START, config.LAYER_FINETUNING_END, config.LAYER_FINETUNING_STEP):
        
        
        finetune_save_path = model_save_path[:-1]  + '_finetuning' + str(layer_index) + '/'
        
        survmodel.set_base_trainable(from_layer=layer_index, to_layer=None)
        
        optimizer = config.get_finetuning_optimizer(config.LR_FINETUNING)
        
        survmodel.compile(optimizer=optimizer,
                          loss = loss)
        survmodel.fit(dh, epochs=config.EPOCHS_2, 
                      steps_per_epoch=config.STEPS_PER_EPOCH,
                      workers=config.WORKERS,
                      max_queue_size=config.MAX_QUEUE_SIZE)
        
        survmodel.save(finetune_save_path)
                        
                       
        config.LR_FINETUNING = config.LR_FINETUNING*config.FINETUNING_LR_FACTOR
        
    
if __name__=='__main__':
    main()