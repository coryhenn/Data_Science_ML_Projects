import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class InputData():
    def __init__(self):
        self.features = None
        self.targets = None

    def readfromFile(self, filename, type="train"):
        if not os.path.exists(filename):
            print("{} file does not exits.".format(filename))
            sys.exit()
        print("Data File: ", filename)
        if type == "train":
            # reading dataset from csv
            df = pd.read_csv(filename)
            # output only values without headers
            X = df.iloc[:, :-1].values
            y = df.target.values
        else:  # Test data
            # reading dataset from csv
            df = pd.read_csv(filename)
            # output only values without headers
            X = df.iloc[:, :-1].values
            y = df.id.values
        return df, X, y

    def inputInfo(self, df):
        print(df.head())
        print(df.info())
        print(df.describe())

    def targetClasses(self, y):
        # Find the number of classes in the target column/data
        classes, _ = np.unique(y, return_counts=True)
        class_count = len(classes)
        print("Classes:", classes, "Class Count:", class_count)
        return class_count

    # Standardize only X data.
    def standardizeFeatures(self, df):
        # standardize , without Standardize , exponent of a big value can overflow to infinity
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
        return X

    def splitDataset(self, X, y):
        # spliting of dataset into train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
        return X_train, X_test, y_train, y_test

    def toDataLoader(self, X, y, batch_size=20):
        torch.manual_seed(0)
        # Convert the data into PyTorch tensors
        #print("Converting data to tensor ....")
        X_tensor = torch.Tensor(X).float()
        y_tensor = torch.Tensor(y).long()
        dataSet = TensorDataset(X_tensor, y_tensor)
        dataLoader = DataLoader(dataSet, batch_size,
                                shuffle=False, num_workers=1)
        return dataLoader

    def X2Tensor(self, X):
        torch.manual_seed(0)
        # Convert the data into PyTorch tensors
        #print("Converting data to tensor ....")
        X_tensor = torch.Tensor(X).float()
        return X_tensor
