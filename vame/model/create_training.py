#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:30:44 2019

@author: luxemk
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal

from vame.util.auxiliary import read_config
    

def temporal_traindata(cfg, files, testfraction, num_features, egocentric):
    
    if egocentric:
        X_train = []
        for file in files:
            path_to_file=cfg['project_path']+'data/'+file+'/'+file+'-PE-seq.npy'
            X_temp = np.load(path_to_file)
            X_train.append(X_temp)
        
        X = np.concatenate(X_train, axis=1)
        X_mean = np.nanmean(X)
        X_std = np.nanstd(X)        
        X = (X-X_mean) / X_std
       
        num_frames = len(X.T)
        test = int(num_frames*cfg['TestFraction'])
        
        z_test =X[:,:test]
        z_train = X[:,test:]    
        
        np.save(cfg['project_path']+'data/train/sequence_train_full_t.npy', z_train)
        np.save(cfg['project_path']+'data/train/sequence_test_full_t.npy', z_test)
        
    else:
        X_train = []
        pos = []
        pos_temp = 0
        pos.append(0)
        for file in files:
            path_to_file=cfg['project_path']+'data/'+file+'/'+file+'-PE-seq.npy'
            X = np.load(path_to_file)
            X_len = len(X.T)
            pos_temp += X_len
            pos.append(pos_temp)
            X_train.append(X)
        
        X = np.concatenate(X_train, axis=1)
        
        seq_inter = np.zeros((X.shape[0],X.shape[1]))
        for s in range(num_features):
            seq_temp = X[s,:]
            seq_pd = pd.Series(seq_temp)  
            seq_pd_inter = seq_pd.interpolate(method="linear", order=None)
            seq_inter[s,:] = seq_pd_inter
        
        X_med = scipy.signal.savgol_filter(seq_inter, 5, 2)
        num_frames = len(X_med.T)
        test = int(num_frames*cfg['TestFraction'])
        
        z_test =X_med[:,:test]
        z_train = X_med[:,test:]    
        
        np.save(cfg['project_path']+'data/train/train_seq.npy', z_train)
        np.save(cfg['project_path']+'data/train/test_seq.npy', z_test)
        
        for i, file in enumerate(files):
            np.save(cfg['project_path']+'data/'+file+'/'+file+'-PE-seq-clean.npy', X_med[:,pos[i]:pos[i+1]])
        
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))
    
def create_trainset(config, egocentric=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)    
    
    path_to_file = cfg['project_path']+'data/'
    if not os.path.exists(cfg['project_path']+'data/train/'):
        os.mkdir(path_to_file+'train/')
    
    files = []     
    for file in cfg['video_sets']:
        use_file = input("Do you want to train on " + file + "? yes/no: ")
        if use_file == 'yes':
            files.append(file)
        if use_file == 'no':
            continue
        
    print("Creating training dataset.")
    temporal_traindata(cfg, files, cfg['TestFraction'], cfg['num_features'], egocentric)
