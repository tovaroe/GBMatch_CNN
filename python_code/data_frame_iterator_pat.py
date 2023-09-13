# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:28:11 2020

@author: Tom
"""

from keras_preprocessing.image import DataFrameIterator
import numpy as np

class DataFrameIteratorPat(DataFrameIterator):
    
    def __init__(self, dataframe, patid_col, *args, **kwargs):
        super().__init__(dataframe, *args, **kwargs)
        self._patids = self.df[patid_col].values
        self.n = len(np.unique(self._patids))
        
    def _filter_valid_filepaths(self, df, x_col):
        self.df = super()._filter_valid_filepaths(df, x_col)
        return self.df
    
    def _set_index_array(self):
        unique_ids = np.unique(self._patids)
        self.index_array = -np.ones_like(unique_ids, dtype='int64')
        for i, unique_id in enumerate(unique_ids):
            indices = np.where(self._patids==unique_id)[0]
            self.index_array[i] = np.random.choice(indices, 1)
            
        self.n = len(self.index_array)
        
        if self.shuffle:
            self.index_array = np.random.permutation(self.index_array)
        