import tensorflow as tf
from tensorflow import keras as k
tf.config.list_physical_devices()
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1' # needed due to dll bug in my environment - might be unnecessary for other ppl

from survival_model import SurvivalModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import sys 
sys.path.append(r'..\..\WSIHandler') # really I should just finally package this...

from wsi_handler import WSIHandler

class GBMPredictor():
    def __init__(self):
        print("Initializing GBMPredictor...\nLoading OS model...")
        
        path_os = os.path.dirname(__file__) + r'\trained_model_weights\OS\\' 
        self.model_os = SurvivalModel.load(path_os, target='OS')
        
        print('Loading TS model...')
        path_ts = os.path.dirname(__file__) + r'\trained_model_weights\TS\\' 
        self.model_ts = SurvivalModel.load(path_ts, target='TS')
        
        self._StandardScaler = self._load_StandardScaler()
        
    def _load_StandardScaler(self):
        
        stds = StandardScaler()
        stds.scale_ =  0.53111313 #0.48140656
        stds.mean_ = 0.38775175# 0.07417872 
        stds.var_ = 0.28208116 #0.23175227
        return stds
        
    def predict(self, path, batch_size=64, annotation_handling = 'exclude'):
        output_path = path.split('.')[0] + '/'
        self.wsi = WSIHandler(path)
        self.wsi.obtain_tissue_mask(annotation_handling=annotation_handling)
        
        tile_gen = self.wsi.tile_generator(width = 512, height = 512)
        self.df = pd.DataFrame()
        done = False
        
        while not done:
            batch_img = []
            batch_coords = []
            
            for i in range(batch_size):
                try:
                    img, coords = next(tile_gen)

                    batch_img.append(np.array(img))
                    batch_coords.append(coords)
                except StopIteration:
                    done = True
                    break
            
            if not batch_coords:
                break
             
            batch_y_pred = self.model_os.model.predict(np.array(batch_img))
            batch_TS_pred = self.model_ts.model.predict(np.array(batch_img))    
   
            df_toadd = pd.DataFrame({
                'x': np.array(batch_coords)[:,0],
                'y': np.array(batch_coords)[:,1],
                'y_pred': batch_y_pred.squeeze(),
                'mesenchymal_pred': batch_TS_pred[:,0].squeeze(),
                'proneural_pred': batch_TS_pred[:,1].squeeze(),
                'classical_pred': batch_TS_pred[:,2].squeeze()
            })
            
            self.df = self.df.append(df_toadd).reset_index(drop=True)
        
        # identify predicted TS
        predicted_TS = self.df[['mesenchymal_pred', 'proneural_pred', 'classical_pred']].mean(axis=0).idxmax(axis=1)[:-5]
        
        # correctly scale results
        self.df['y_pred_scaled'] = self._StandardScaler.transform(self.df[['y_pred']]) 
        self.df['high_risk'] = self.df['y_pred_scaled'] > 1
        risk_group = 'High risk' if self.df.high_risk.mean() >= 0.25 else 'Low risk'
           
        # make heatmaps
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        ## ts heatmap
        arr = np.empty_like(self.wsi.tissue_mask_tile_generator, dtype=np.float32)
        arr[:] = np.NaN

        f, axs = plt.subplots(1,3, figsize=(20,6))
        cols = ['mesenchymal_pred', 'classical_pred', 'proneural_pred']

        for j, ts in enumerate(cols):  
            for i in self.df.index:
                x = self.df.loc[i, 'x']
                y = self.df.loc[i, 'y']
                TS_pred = self.df.loc[i, ts]

                arr[x,y] = TS_pred

            mean = str(np.round(self.df[ts].mean(), 2))
            
            ax = axs[j]
            sns.heatmap(arr, xticklabels=False, yticklabels=False, square=True, vmin=0, vmax=1, cmap='icefire', ax=ax)
            ax.set_title('TS-Score: ' + ts.split('_')[0] + '\n(mean score ' + mean + ')')
            
        plt.suptitle('Predicted TS: ' + predicted_TS) 
        plt.savefig(output_path+'ts_heatmap.png', bbox_inches='tight')
        plt.close()
        
        ## survival heatmap
        arr = np.empty_like(self.wsi.tissue_mask_tile_generator, dtype=np.float32)
        arr[:] = np.NaN

        for i in self.df.index:
                x = self.df.loc[i, 'x']
                y = self.df.loc[i, 'y']
                y_pred= self.df.loc[i, 'y_pred_scaled']

                arr[x,y] = y_pred

        sns.heatmap(arr, xticklabels=False, yticklabels=False, square=True, cmap='icefire', vmin=-2, vmax=2)
        plt.title('Median risk score: '+ str(np.round(self.df.y_pred_scaled.median(), 2)) + '\n' + risk_group + ' (' + str(np.round(self.df.high_risk.mean()*100,1)) + '% high risk tiles)')
        plt.savefig(output_path+'risk_heatmap.png', bbox_inches='tight')
        plt.close()
        
        self.df.to_csv(output_path+'results_per_tile.csv')
        
        return (predicted_TS, risk_group, self.df['y_pred_scaled'].median())

        