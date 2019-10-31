import numpy as np 

acc_data = np.load('acc_data.npy')
gyro_data = np.load('gyro_data.npy')

print(acc_data.shape, gyro_data.shape)