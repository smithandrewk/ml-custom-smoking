import datetime
from lib.env import DATA_PATH
import os
import pandas as pd
from torch.utils.data import TensorDataset
import json
import torch

def datetime_to_epoch(year, month, day, hour, minute, second):
    # Create the datetime object
    dt = datetime.datetime(year, month, day, hour, minute, second)
    
    # Subtract 4 hours (to convert from EST to UTC)
    dt_utc = dt - datetime.timedelta(hours=4)
    
    # Convert to epoch time
    epoch_time = int(dt_utc.timestamp())
    
    # Return nanoseconds as requested
    return epoch_time * 1e9

def get_dataset_for_project(project,windowsize,stride=100,balance=True):
    labels_path = f'{DATA_PATH}/1_labeled/{project}/labels.json'
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    regions = os.listdir(f'{DATA_PATH}/2_regions/{project}')

    X_train = []
    y_train = []

    for region in regions:
        df = pd.read_csv(f'{DATA_PATH}/2_regions/{project}/{region}')
        df.timestamp = df.timestamp.astype('datetime64[ns]')
        region_labels = [(datetime.datetime.strptime(label['start'], '%Y-%m-%d %H:%M:%S.%f'),datetime.datetime.strptime(label['end'], '%Y-%m-%d %H:%M:%S.%f')) for label in labels]
        region_labels = [label for label in region_labels if ((label[0] > df.timestamp.min()) & (label[1] < df.timestamp.max()))]
        df['y_true'] = 0
        for label in region_labels:
            df.loc[((df.timestamp > label[0]) & (df.timestamp < label[1])),'y_true'] = 1
        # df_resampled = df.set_index('timestamp').resample('20ms').mean().reset_index()
        df_resampled = df.copy()
        X = torch.from_numpy(df_resampled[['x','y','z']].values).float()
        y = torch.from_numpy(df_resampled['y_true'].values).float()
        for i in range(0,len(X) - windowsize,stride):
            X_train.append(X[i:i+windowsize])
            y_train.append(y[i + (windowsize // 2)])
    X_train = torch.stack(X_train).transpose(1,2)
    y_train = torch.tensor(y_train).reshape(-1,1).float()
    if balance:
        idx_0 = torch.where(y_train == 0)[0]
        idx_0 = idx_0[torch.randperm(len(idx_0))[:torch.bincount(y_train.flatten().long())[1]]]
        idx_1 = torch.where(y_train == 1)[0]
        idx = torch.cat([idx_0,idx_1])
        X_train,y_train = X_train[idx],y_train[idx]

    return TensorDataset(X_train,y_train)