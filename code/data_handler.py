# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:58:54 2020

@author: Tom
"""

from tensorflow import keras as k
import pandas as pd
import numpy as np
import preprocess_crop #courtesy of Rustam Aliyev
import matplotlib.pyplot as plt
from image_data_generator_pat import ImageDataGeneratorPat
from negative_log_likelihood import NegativeLogLikelihood
from c_index_callback import CIndexCallback
from sksurv.metrics import concordance_index_censored as cindex
import lifelines as ll
from ast import literal_eval
import os
import config


from randaugment import distort_image_with_randaugment


class DataHandler(object):
    def __init__(self,
                 image_path,
                 annotation_path,
                 magnification = 'x20',
                 im_format = 'jpeg',
                 y_col='idx',
                 im_size = 512,
                 batch_size = config.BATCH_SIZE,
                 rescale=None,
                 weight_col = None,
                 class_mode='raw',
                 interpolation='lanczos:random',
                 evaluation=False,
                 kfold_val=None,
                 **kwargs):
        
        
        self._IMAGE_PATH = image_path
        self._MAGNIFICATION = magnification 
        self._IM_FORMAT = im_format

        self._RESCALE = rescale
        self._Y_COL = y_col
        self._IM_SIZE = im_size
        self._CLASS_MODE = class_mode
        self._INTERPOLATION = interpolation  
        self._KFOLD_VAL = kfold_val
        self._WEIGHT_COL = None
        
        self.annotation = self.read_annotation(annotation_path)
        (self.df_train, self.df_validation, self.df_test) = self.train_test_split()
        
        self._BATCH_SIZE = batch_size
        
        self.evaluation_setup = False
        if evaluation:
            self.setup_evaluation_generator()
        else:
            self._setup_generator(**kwargs)       

        
    def _setup_generator(self, **kwargs):
            
        preprocessing_function = lambda image: distort_image_with_randaugment(image.astype(np.uint8), 3, 5, 'Default')
        
        self.train_datagen = ImageDataGeneratorPat(
            preprocessing_function=preprocessing_function,
            rescale=self._RESCALE,
            **kwargs)
        
        self.test_datagen = ImageDataGeneratorPat(
            rescale=self._RESCALE)
            

            
        print('Setting up validation generator...')
        self.val_generator = self.test_datagen.flow_from_dataframe(
            self.df_validation,
            y_col=self._Y_COL,
            target_size=(self._IM_SIZE, self._IM_SIZE),
            class_mode=self._CLASS_MODE,
            interpolation='lanczos:center',
            batch_size=self._BATCH_SIZE)
        

        print('Setting up train generator...')
        print(f'Weight col: {self._WEIGHT_COL}')
        self.train_generator = self.train_datagen.flow_from_dataframe(
            self.df_train,
            y_col=self._Y_COL,
            weight_col=self._WEIGHT_COL,
            target_size=(self._IM_SIZE, self._IM_SIZE),
            class_mode=self._CLASS_MODE,
            interpolation=self._INTERPOLATION,
            batch_size=self._BATCH_SIZE)
                
        
    def setup_evaluation_generator(self):
        self.datagen = k.preprocessing.image.ImageDataGenerator(  
            rescale=self._RESCALE)
        
        print('Setting up validation evaluation generator...')
        self.val_generator = self.datagen.flow_from_dataframe(
            self.df_validation,
            y_col=self._Y_COL,
            target_size=(self._IM_SIZE, self._IM_SIZE),
            class_mode=self._CLASS_MODE,
            interpolation='lanczos:center',
            batch_size=self._BATCH_SIZE,
            shuffle=False)

        
        print('Setting up train evaluation generator...')
        self.train_generator = self.datagen.flow_from_dataframe(
            self.df_train,
            y_col=self._Y_COL,
            target_size=(self._IM_SIZE, self._IM_SIZE),
            class_mode=self._CLASS_MODE,
            interpolation='lanczos:center',
            batch_size=self._BATCH_SIZE,
            shuffle=False)
        
        self.evaluation_setup = True
        
    def show_images(self, gen='train'):
        if gen=='train':
            images, y = next(self.train_generator)
        elif gen=='val':
            images, y = next(self.val_generator)
        elif gen=='test':
            images, y = next(self.test_generator)
        
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(images[i]/255) # only works if rescale = None
            plt.title(y[i])
            plt.axis('off')
        plt.show()
            
    def read_annotation(self, annotation_path):
        df = pd.read_csv(annotation_path, index_col=0)
        df.viable_ids = df.viable_ids.apply(lambda l: literal_eval(l)) 
        df = df.loc[df.surgery==1]
        df['surgery'] = df.surgery.astype(int)
        
        drop_subset = ['patID', 'surgery', 'Age', 'Sex', 
                       'image_folder_name', 'viable_ids']
        if self._Y_COL == 'idx':
            drop_subset = drop_subset + ['followup_years', 'VitalStatus']
        elif 'Mesenchymal' in self._Y_COL:
            drop_subset = drop_subset + ['Mesenchymal', 'Classical', 'Proneural']
        df.dropna(subset=drop_subset, inplace=True)
        

    def train_test_split(self, val_size=0.2, random_state=74):
        df = self.annotation   
        
        trainval_id = df.index
        

        validation_id = df[df.KFold_val == self._KFOLD_VAL].index
        train_id = trainval_id.drop(validation_id)
    
        # introduce sample weights
        df_train_pat = df.loc[train_id,:]
        if 'Mesenchymal' in self._Y_COL:
            df_train_pat = df.loc[train_id,:]
            A = df_train_pat[['Mesenchymal', 'Classical', 'Proneural']].to_numpy().T
            b = np.ones(3)*len(df_train_pat)*1/3
            x = np.linalg.lstsq(A,b, rcond=None)
            df_train_pat['sample_weight'] = x[0]
            self._WEIGHT_COL = 'sample_weight'
                  
        df_train = df_train_pat.explode('filenames').rename({'filenames': 'filename'}, axis=1)
        
        
        df_validation = df.loc[validation_id,:].explode('filenames').rename({'filenames': 'filename'}, axis=1)
        df_test = pd.DataFrame
        
        
        return (df_train, df_validation, df_test)        
        
        # set filenames
        def get_filenames(patID):
            folder = df.loc[df.patID==patID, 'image_folder_name'].item()
            filenames = []
            folder_path = self._IMAGE_PATH + folder      
        
            
            for _, _, files in os.walk(folder_path):
                f = files[0]
                if not self._IM_FORMAT in f: f = files[1]
                if not self._IM_FORMAT in f: raise ValueError(f"Couldn't get filenames for {folder}")
                filename_trunk = '_'.join(f.split(sep='.')[0].split('_')[:-1])  + '_'               
            
            for i in df.loc[df.patID==patID, 'viable_ids'].item():   
                filenames.append(self._IMAGE_PATH + folder + '/' + filename_trunk + str(i) + '.' + self._IM_FORMAT)
                
            return filenames
        
        df['filenames'] = df.patID.apply(lambda patid: get_filenames(patid))
                    
        df.loc[:,'LTS_median'] = (df['followup_years'] > df['followup_years'].median()).astype(str)
        df.loc[:,'LTS_3'] = (df['followup_years'] > 3).astype(str)
        df.loc[(df['VitalStatus'] == 'alive') & (df['LTS_median'] == 'False'), 'LTS_median'] = np.nan
        
        df.reset_index(drop=True, inplace=True)
        df['idx'] = df.index
        
        return df
    
            
    def get_nll_function(self):
        # works only when y_col is accordingly sorted index!
        followup = self.annotation['followup_years']
        event = self.annotation['VitalStatus'].map(dict(dead=1, alive=0, lost=0))
        nll = NegativeLogLikelihood(followup, event)
        return nll.loss
    
    def get_cindex_callback(self, loss_func=None, validation_rounds=1):
        validation_ids = self.df_validation.idx.unique()
        
        followup = self.annotation.loc[validation_ids, 'followup_years']
        event = self.annotation.loc[validation_ids, 'VitalStatus'].map(dict(dead=1, alive=0, lost=0))
        return CIndexCallback(self.val_generator, validation_ids, followup, event, 
                              loss_func = loss_func,
                              validation_rounds = validation_rounds)
    
    def predict_samples(self, survmodel, cohort='all', savepath=None):
        
        assert self.evaluation_setup, 'Please run setup_evaluation_generator first.'
       
        if not cohort in ['all', 'train', 'test', 'val']: raise ValueError('\'cohort\' must be one of [\'all\', \'train\', \'val\', \'test\'], got {}'.format(cohort))
       
        col_dict = {'idx': 'y_pred',
                    'Mesenchymal': 'Mesenchymal_pred',
                    'Proneural': 'Proneural_pred',
                    'Classical': 'Classical_pred'}
       
        if self._Y_COL == 'idx':
            pred_cols = ['y_pred']
        else:
            pred_cols = list(map(lambda s: col_dict[s], list(self._Y_COL)))
       
        if (cohort=='train' or cohort=='all'):
            df_toadd = pd.DataFrame(survmodel.model.predict(self.train_generator), columns = pred_cols, index=self.df_train.index)
            self.df_train[pred_cols] = df_toadd
            if not savepath==None:
                self.df_train.to_csv(savepath + '/df_train.csv')
        
        if (cohort=='val' or cohort=='all'):
            df_toadd = pd.DataFrame(survmodel.model.predict(self.val_generator), columns = pred_cols, index=self.df_validation.index)
            self.df_validation[pred_cols] = df_toadd
            if not savepath==None:
                self.df_validation.to_csv(savepath + '/df_validation.csv')

    def predict_samples_features(self, survmodel, savepath):
        
        assert self.evaluation_setup, 'Please run setup_evaluation_generator first.'
       

        survmodel_trunc = k.models.Model(inputs = survmodel.model.input, outputs = survmodel.model.get_layer(index=5).output)
       
       

        df_1 = pd.DataFrame(survmodel_trunc.predict(self.train_generator), index=self.df_train.index)
        df_1['training'] = 1
        df_1['Age'] = self.df_train['Age']
        df_1['Sex'] = self.df_train['Sex']
        df_1['followup_years'] = self.df_train['followup_years']
        df_1['VitalStatus'] = self.df_train['VitalStatus']
        df_1['filename'] = self.df_train['filename']
        df_1['LTS_median'] = self.df_train['LTS_median']
        df_1['LTS_3'] = self.df_train['LTS_3']
        df_1['Proneural'] = self.df_train['Proneural']
        df_1['Classical'] = self.df_train['Classical']
        df_1['Mesenchymal'] = self.df_train['Mesenchymal']
        
        df_2 = pd.DataFrame(survmodel_trunc.predict(self.val_generator), index=self.df_validation.index)
        df_2['training'] = 0
        df_2['Age'] = self.df_validation['Age']
        df_2['Sex'] = self.df_validation['Sex']
        df_2['followup_years'] = self.df_validation['followup_years']
        df_2['VitalStatus'] = self.df_validation['VitalStatus']
        df_2['filename'] = self.df_validation['filename']
        df_2['LTS_median'] = self.df_validation['LTS_median']
        df_2['LTS_3'] = self.df_validation['LTS_3']
        df_2['Proneural'] = self.df_validation['Proneural']
        df_2['Classical'] = self.df_validation['Classical']
        df_2['Mesenchymal'] = self.df_validation['Mesenchymal']
        
        df = df_1.append(df_2)
        
        df.to_csv(savepath + '/df_features.csv')

    def get_cindex(self, func=np.median, cohort='val'):        

        aggfuncs = {'followup_years': 'first',
            'VitalStatus': 'first',
            'y_pred': func}
        
        if cohort=='train': df = self.df_train
        elif cohort=='val': df = self.df_validation
        
        assert 'y_pred' in df.columns, 'Please run predict_samples first.'
        
        df_grouped = df.groupby('patID')[['followup_years', 'VitalStatus', 'y_pred']].agg(aggfuncs)
        c = cindex(df_grouped.VitalStatus.map(dict(dead=True, alive=False, lost=False)), df_grouped['followup_years'], df_grouped.y_pred)
        return c
    
    def get_nll(self, func=np.median, cohort='val'):
        aggfuncs = {'idx': 'first',
            'y_pred': func}
        
        if cohort=='train': df = self.df_train
        elif cohort=='val': df = self.df_validation
        
        assert 'y_pred' in df.columns, 'Please run predict_samples first.'
        
        df_grouped = df.groupby('patID')[['idx', 'y_pred']].agg(aggfuncs)
        nll = self.get_nll_function()(df_grouped.idx.tolist(), df_grouped.y_pred.tolist()).numpy()
        return nll
    
    def plot_survival(self, func=np.median, cohorts = ['train', 'val'], save_path=None):
        aggfuncs = {'followup_years': 'first',
            'VitalStatus': 'first',
            'y_pred': func}
        
        for cohort in cohorts:
            if cohort=='train':
                df = self.df_train
            elif cohort=='val':
                df = self.df_validation
            else:
                raise ValueError("Cohort must be one of ['train', 'val', 'test'], got {}.".format(cohort))
                
            assert 'y_pred' in df.columns, 'Please run predict_samples first.' 
                
            df_grouped = df.groupby('patID')[['followup_years', 'VitalStatus', 'y_pred']].agg(aggfuncs)
            df_grouped['pred_high'] = df_grouped.y_pred > df_grouped.y_pred.median()
            
            kmf = ll.KaplanMeierFitter()
            ax = plt.subplot(111, label=cohort)
            ax.clear()
            
            followup_high = df_grouped.loc[df_grouped['pred_high'], 'followup_years']
            event_high = df_grouped.loc[df_grouped['pred_high'], 'VitalStatus'].map(dict(dead=True, alive=False, lost=False))
            followup_low = df_grouped.loc[~df_grouped['pred_high'], 'followup_years']
            event_low = df_grouped.loc[~df_grouped['pred_high'], 'VitalStatus'].map(dict(dead=True, alive=False, lost=False))
            
            kmf.fit(followup_high, 
                    event_high, 
                    label="y_pred_high")
            kmf.plot(ax=ax)
            kmf.fit(followup_low, 
                    event_low, 
                    label="y_pred_low")
            kmf.plot(ax=ax)
            
            p_value = ll.statistics.logrank_test(followup_high, followup_low, event_high, event_low).p_value
            p_value = np.round(p_value, 3)
            ax.annotate("p = " + str(p_value), (3,0.7))
            
            c_index = self.get_cindex(func=func, cohort=cohort)[0]
            c_index = np.round(c_index, 3)
            ax.annotate("c-index = " + str(c_index), (3, 0.6))
            nll = np.round(self.get_nll(func=func, cohort=cohort), 3)
            ax.annotate("loss = " + str(nll), (3, 0.5))
            
            if save_path:
                plt.savefig(save_path + 'KM_' + cohort + '.png')
        
        plt.show()    
        
        ax.clear()
        
        
        