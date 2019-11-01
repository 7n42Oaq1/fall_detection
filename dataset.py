import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset
import os
import random
from torchvision import transforms, utils

'''
Create Pytorch dataset class
'''
class FallDetectionDataset(Dataset):
    def __init__(self, type_='train', transform=None):
        self.type = type_
        self.transform = transform
        self.acc_data, self.gyro_data, self.label = self.get_data()
    
    def get_data(self):
        acc_data = np.load('dataset/' + 'acc_x_' + self.type + '.npy')
        gyro_data = np.load('dataset/' + 'gyro_x_' + self.type + '.npy')
        label = np.load('dataset/' + 'y_' + self.type + '.npy')
        return acc_data, gyro_data, label
    
    def __len__(self):
        return self.acc_data.shape[0]
    
    def __getitem__(self, idx):
        X_acc = self.acc_data[idx]
        X_acc = X_acc.transpose((1, 0))
        X_gyro = self.gyro_data[idx]
        X_gyro = X_gyro.transpose((1, 0))

        X = np.stack((X_acc, X_gyro), axis=0)
        y = self.label[idx]

        if self.transform:
            X = self.transform(X)
        return X, y

class To2D(object):
    def __init__(self, shape):
        self.shape = shape
    def __call__(self, x):
        new_acc = None
        new_gyro = None
        if isinstance(self.shape, int):
            new_acc = x[0].reshape((3,self.shape, self.shape))
            new_gyro = x[1].reshape((3,self.shape, self.shape))
        else:
            new_acc = x[0].reshape((3,self.shape[0], self.shape[1]))
            new_gyro = x[1].reshape((3,self.shape[0], self.shape[1]))
        new_x = np.stack((new_acc, new_gyro), axis=0)
        return new_x            

if __name__ == "__main__":
    compose = transforms.Compose([To2D(20)])
    dataset = FallDetectionDataset(transform=compose)
    X,y = dataset[0]

    dataloader = DataLoader(dataset, 10, False)
    for i, data in enumerate(dataloader):
        X = data[0]
        y = data[1]
        print(X.shape, y.shape) 
        print()