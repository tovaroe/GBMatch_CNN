
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:14:19 2020

@author: Tom
"""

import tensorflow as tf
from tensorflow import keras as k
tf.config.list_physical_devices()
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1' # needed due to dll bug in my environment - might be unnecessary for other ppl
import shutil
import numpy as np
from scipy.stats import mode

import config
from data_handler import DataHandler
from survival_model import SurvivalModel
from ts_callback import TSCallback





def main():

    for i in range(5):    

        kfold_val = i
        model_save_path = config.model_save_path+f'_TS_fold{kfold_val}/'
    
        dh = DataHandler(config.image_path, config.annotation_path, magnification=config.MAGNIFICATION,
                         y_col=['Mesenchymal', 'Proneural', 'Classical'],
                         class_mode=config.CLASS_MODE,
                         im_format=config.IM_FORMAT,
                         im_size=config.IM_SIZE, 
                         rescale=config.RESCALE,
                         batch_size=config.BATCH_SIZE,
                         kfold_val=kfold_val)
        
    
        
        loss = 'mse'
        callbacks = [TSCallback(dh.val_generator, validation_rounds=config.VALIDATION_ROUNDS)]
        
        # Create new model if no old model is found
        if not os.path.exists(model_save_path):
            print("No pre-existing model found, building new model...")
            survmodel = SurvivalModel.build_new(input_shape=(config.IM_SIZE, config.IM_SIZE, 3),
                                                base_model_name=config.BASE_MODEL,
                                                dropout=config.DROPOUT,
                                                target = 'TS')
            survmodel.compile(optimizer=config.OPTIMIZER_1,
                              loss = loss)
            survmodel.fit(dh, epochs=config.EPOCHS_1, 
                          steps_per_epoch=config.STEPS_PER_EPOCH,
                          callbacks = callbacks,
                          workers=config.WORKERS,
                          max_queue_size=config.MAX_QUEUE_SIZE)
            
            survmodel.save(model_save_path)
            
            for filename in os.listdir():
                if filename.endswith('.py'):            
                    shutil.copy(filename, model_save_path+filename.replace('.py', '.txt'))
                    
                    
            dh_eval = DataHandler(config.image_path, config.annotation_path, magnification=config.MAGNIFICATION,
                                 y_col=['Mesenchymal', 'Proneural', 'Classical'],
                                 class_mode=config.CLASS_MODE,
                                 im_format=config.IM_FORMAT,
                                 im_size=config.IM_SIZE, 
                                 rescale=config.RESCALE,
                                 batch_size=config.BATCH_SIZE,
                                 evaluation=True,
                                 kfold_val=kfold_val)
        
            dh_eval.predict_samples(survmodel, cohort='all', savepath=model_save_path)
            
        # Load pre-saved model if it exists in the given path (see config)    
        else:
            model_load_path = model_save_path
            for layer_index in range(config.LAYER_FINETUNING_START, config.LAYER_FINETUNING_END, config.LAYER_FINETUNING_STEP):
                finetune_load_path = model_save_path[:-1]  + '_finetuning' + str(layer_index) + '/'
                if os.path.exists(finetune_load_path):
                    model_load_path = finetune_load_path
                    config.LR_FINETUNING = config.LR_FINETUNING*config.FINETUNING_LR_FACTOR
                else:
                    config.LAYER_FINETUNING_START = layer_index
                    break
            
            print("\n\nLoading model saved at: ", model_load_path,  " ...\n\n")
            survmodel = SurvivalModel.load(model_load_path, target='TS')
            
    
        
        # Perform iterative finetuning if wanted
        for layer_index in range(config.LAYER_FINETUNING_START, config.LAYER_FINETUNING_END, config.LAYER_FINETUNING_STEP):
            
            
            finetune_save_path = model_save_path[:-1]  + '_finetuning' + str(layer_index) + '/'
            
            survmodel.set_base_trainable(from_layer=layer_index, to_layer=None)
            
            optimizer = config.get_finetuning_optimizer(config.LR_FINETUNING)
            
            survmodel.compile(optimizer=optimizer,
                              loss = loss)
            survmodel.fit(dh, 
                          epochs=config.EPOCHS_2, 
                          steps_per_epoch=config.STEPS_PER_EPOCH,
                          callbacks = callbacks,
                          
                          workers=config.WORKERS,
                          max_queue_size=config.MAX_QUEUE_SIZE)
            
            survmodel.save(finetune_save_path)
            
            for filename in os.listdir():
                if filename.endswith('.py'):            
                    shutil.copy(filename, finetune_save_path+filename.replace('.py', '.txt'))
                    
                    
            dh_eval = DataHandler(config.image_path, config.annotation_path, magnification=config.MAGNIFICATION,
                                 y_col=['Mesenchymal', 'Proneural', 'Classical'],
                                 class_mode=config.CLASS_MODE,
                                 im_format=config.IM_FORMAT,
                                 im_size=config.IM_SIZE, 
                                 rescale=config.RESCALE,
                                 batch_size=config.BATCH_SIZE,
                                 evaluation=True,
                                 kfold_val=kfold_val)
        
            dh_eval.predict_samples(survmodel, cohort='all', savepath=finetune_save_path)
            
            config.LR_FINETUNING = config.LR_FINETUNING*config.FINETUNING_LR_FACTOR
        
    
if __name__=='__main__':
    main()