# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:21:17 2021

@author: Tom
"""


import sys


from tensorflow import keras as k

# Dependency - change to corret path before use!
randaugment_path =  r'../../pathology-he-auto-augment/he-randaugment' 
sys.path.append(randaugment_path)

# image variables
IM_FORMAT = 'jpeg'
RESCALE = None # should be okay if using preprocess_input in survival_model
MAGNIFICATION = 'x20'
BASE_MODEL = 'Xception'
IM_SIZE = 512 

# path variables
image_path = '../data/training/image_tiles/'
annotation_path = '../data/training/GBMatch_annotation.csv'
model_save_path = '../data/training/models/' + BASE_MODEL + '_' + MAGNIFICATION + '_' + str(IM_SIZE)

# data variables
CLASS_MODE = 'raw'
oversample_LTS = False
BATCH_SIZE = 64

# (pre-)processing variables
WORKERS = 6 
MAX_QUEUE_SIZE = 12    
STEPS_PER_EPOCH = 150 # for manuscript: 150
VALIDATION_ROUNDS = 20 # for manuscript: 20
EPOCHS_1 = 25 # for manuscript: 25
EPOCHS_2 = 10 # for manuscript: 10

# DL variables    
DROPOUT = 0.25
lr_schedule = k.optimizers.schedules.ExponentialDecay(0.001,
                                                      decay_steps = 400,
                                                      decay_rate=0.9,
                                                      staircase=True)
OPTIMIZER_1 = k.optimizers.Adam(learning_rate=lr_schedule,
                                epsilon=1e-2)         

LR_FINETUNING = 0.0001
FINETUNING_LR_FACTOR = 0.9
get_finetuning_optimizer = lambda lr: k.optimizers.Adam(learning_rate=k.optimizers.schedules.ExponentialDecay(lr,
                                                                                                   decay_steps=400,
                                                                                                   decay_rate=0.8,
                                                                                                   staircase=True))
# Finetuning variables
LAYER_FINETUNING_START = 125
LAYER_FINETUNING_END = 123
LAYER_FINETUNING_STEP = -5