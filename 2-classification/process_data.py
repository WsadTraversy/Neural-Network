import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from collections import Counter

def get_data(test_dataset=False):
    if not test_dataset:
        df = pd.read_csv('data/train_data.csv', delimiter=',')
    else:
        df = pd.read_csv('data/test_data.csv', delimiter=',')

    df.drop(columns=['SubwayStation'], inplace=True)

    df = df.apply(lambda col: col.astype(float) if col.dtype == 'int' else col, axis=0)

    # TimeToBusStop, TimeToSubway
    unique_values = {'no_bus_stop_nearby':float(0), '0-5min':float(0.25),'0~5min':float(0.25),  '5min-10min':float(0.5),'5min~10min':float(0.5),
                      '10min-15min':float(0.75),'10min~15min':float(0.75), '15min-20min':float(1),'15min~20min':float(1)}

    df['TimeToBusStop'] = df['TimeToBusStop'].map(unique_values)
    df['TimeToSubway'] = df['TimeToSubway'].map(unique_values)

    # 'HallwayType', 'HeatingType', 'AptManageType'
    categorical_columns = ['HallwayType', 'HeatingType', 'AptManageType']
    categorical_values = pd.get_dummies(df[categorical_columns])

    df.drop(columns=categorical_columns, inplace=True)
    
    if not test_dataset:
        train_indices = np.random.rand(len(df))>0.3

        # Targets
        targets_values = df['SalePrice'].apply(lambda row: 0 if row <= 100000.0 else 1 if 100000 < row <= 350000 else 2)

        targets_train = torch.tensor(targets_values.values[train_indices])
        targets_valid = torch.tensor(targets_values.values[~train_indices])
        
        df.drop(columns=['SalePrice'], inplace=True)

        df = (df - df.min()) / (df.max() - df.min())

        numerical_data_train = torch.from_numpy(df.values[train_indices, :]).float()
        numerical_data_valid = torch.from_numpy(df.values[~train_indices, :]).float()
        categorical_data_train = torch.from_numpy(categorical_values.values[train_indices]).float()
        categorical_data_valid = torch.from_numpy(categorical_values.values[~train_indices]).float()

    
        train_dataset = data.TensorDataset(numerical_data_train, categorical_data_train, targets_train)
        validation_dataset = data.TensorDataset(numerical_data_valid, categorical_data_valid, targets_valid)

        return train_dataset, validation_dataset
    else:
        df = (df - df.min()) / (df.max() - df.min())

        numerical_data = torch.from_numpy(df.values).float()
        categorical_data = torch.from_numpy(categorical_values.values).float()

        dataset = data.TensorDataset(numerical_data, categorical_data)

        return dataset
