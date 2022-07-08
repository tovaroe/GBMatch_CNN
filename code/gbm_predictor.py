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
        print("Initializing GBMPredictor...\nLoading OS models...")
                
        paths_os = [os.path.dirname(__file__) + r'\trained_model_weights' + '\OS_fold' + str(i) + '\\' for i in range(5)]
        self.models_os = [SurvivalModel.load(path, target='OS') for path in paths_os]
        
        paths_ts = [os.path.dirname(__file__) + r'\trained_model_weights' + '\TS_fold' + str(i) + '\\' for i in range(5)]
        print("Loading TS models...")
        self.models_ts = [SurvivalModel.load(path, target='TS') for path in paths_ts]
        
        self._StandardScalers = [self._load_StandardScaler(i) for i in range(5)]
        
    def _load_StandardScaler(self, idx):
        stds_attributes_path = os.path.dirname(__file__) + r'\trained_model_weights\StandardScaler_OS_attributes.csv'
        stds_attributes = pd.read_csv(stds_attributes_path, index_col=0)
        
        stds = StandardScaler()
        stds.scale_ = stds_attributes.loc[idx, 'scale']
        stds.mean_ = stds_attributes.loc[idx, 'mean']
        stds.var_ = stds_attributes.loc[idx, 'var']
        return stds
        
    def predict(self, path, batch_size=64):
        output_path = path.split('.')[0] + '/'
        wsi = WSIHandler(path)
        wsi.obtain_tissue_mask()
        
        tile_gen = wsi.tile_generator(width = 512, height = 512)
        df = pd.DataFrame(columns=['x', 'y', 'y_pred_0', 'y_pred_1', 'y_pred_2', 'y_pred_3', 'y_pred_4'])
        done = False
        
        while not done:
            batch_img = []
            batch_coords = []
            batch_y_pred = []
            batch_TS_pred = []
            
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
            
            for model in self.models_os:
                batch_y_pred.append(model.model.predict(np.array(batch_img)))
                
            for model in self.models_ts:
                batch_TS_pred.append(model.model.predict(np.array(batch_img)))
                
                
            df_toadd = pd.DataFrame({
                'x': np.array(batch_coords)[:,0],
                'y': np.array(batch_coords)[:,1],
                'y_pred_0': batch_y_pred[0].squeeze(),
                'y_pred_1': batch_y_pred[1].squeeze(),
                'y_pred_2': batch_y_pred[2].squeeze(),
                'y_pred_3': batch_y_pred[3].squeeze(),
                'y_pred_4': batch_y_pred[4].squeeze(),
                'mesenchymal_pred_0': batch_TS_pred[0][:,0].squeeze(),
                'proneural_pred_0': batch_TS_pred[0][:,1].squeeze(),
                'classical_pred_0': batch_TS_pred[0][:,2].squeeze(),
                'mesenchymal_pred_1': batch_TS_pred[1][:,0].squeeze(),
                'proneural_pred_1': batch_TS_pred[1][:,1].squeeze(),
                'classical_pred_1': batch_TS_pred[1][:,2].squeeze(),
                'mesenchymal_pred_2': batch_TS_pred[2][:,0].squeeze(),
                'proneural_pred_2': batch_TS_pred[2][:,1].squeeze(),
                'classical_pred_2': batch_TS_pred[2][:,2].squeeze(),
                'mesenchymal_pred_3': batch_TS_pred[3][:,0].squeeze(),
                'proneural_pred_3': batch_TS_pred[3][:,1].squeeze(),
                'classical_pred_3': batch_TS_pred[3][:,2].squeeze(),
                'mesenchymal_pred_4': batch_TS_pred[4][:,0].squeeze(),
                'proneural_pred_4': batch_TS_pred[4][:,1].squeeze(),
                'classical_pred_4': batch_TS_pred[4][:,2].squeeze()
            })
            
            df = df.append(df_toadd).reset_index(drop=True)
        
        # average ts predictions
        for ts in ['mesenchymal_pred', 'proneural_pred', 'classical_pred']:
            df[ts+'_mean'] = df[[ts+'_0', ts+'_1', ts+'_2', ts+'_3', ts+'_4']].mean(axis=1)
        
        predicted_TS = df[['mesenchymal_pred_mean', 'proneural_pred_mean', 'classical_pred_mean']].mean(axis=0).idxmax(axis=1)[:-10]
        
        # correctly scale results
        for i in range(5):
            df['y_pred_' + str(i) + '_scaled'] = self._StandardScalers[i].transform(df[['y_pred_'+str(i)]])
           
        df['y_pred_mean'] = df[['y_pred_0_scaled','y_pred_1_scaled','y_pred_2_scaled','y_pred_3_scaled','y_pred_4_scaled']].mean(axis=1)
        df['high_risk'] = df['y_pred_mean'] > 1
        risk_group = 'High risk' if df.high_risk.mean() >= 0.25 else 'Low risk'
           
        # make heatmaps
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        ## ts heatmap
        arr = np.empty_like(wsi.tissue_mask_tile_generator)
        arr[:] = np.NaN

        f, axs = plt.subplots(1,3, figsize=(20,6))

        cols = ['mesenchymal_pred_mean', 'classical_pred_mean', 'proneural_pred_mean']

        for j, ts in enumerate(cols):  
            for i in df.index:
                x = df.loc[i, 'x']
                y = df.loc[i, 'y']
                TS_pred = df.loc[i, ts]

                arr[x,y] = TS_pred

            mean = str(np.round(df[ts].mean(), 2))
            
            ax = axs[j]
            sns.heatmap(arr, xticklabels=False, yticklabels=False, square=True, vmin=0, vmax=1, cmap='icefire', ax=ax)
            ax.set_title('Predicted TS: ' + ts.split('_')[0] + '\n(mean score ' + mean + ')')
            
        plt.savefig(output_path+'ts_heatmap.png', bbox_inches='tight')
        plt.close()
        
        ## survival heatmap
        arr = np.empty_like(wsi.tissue_mask_tile_generator)
        arr[:] = np.NaN

        for i in df.index:
                x = df.loc[i, 'x']
                y = df.loc[i, 'y']
                y_pred= df.loc[i, 'y_pred_mean']

                arr[x,y] = y_pred

        sns.heatmap(arr, xticklabels=False, yticklabels=False, square=True, cmap='icefire', vmin=-2, vmax=2)
        plt.title('Median risk score: '+ str(np.round(df.y_pred_mean.median(), 2)) + '\n' + risk_group + ' (' + str(np.round(df.high_risk.mean()*100,1)) + '% high risk tiles)')
        plt.savefig(output_path+'risk_heatmap.png', bbox_inches='tight')
        plt.close()
        
        df.to_csv(output_path+'results_per_tile.csv')
        
        return (predicted_TS, risk_group)

        