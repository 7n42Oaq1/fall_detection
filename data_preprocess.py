import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pandas as pd

def load_data(resampling_rate=5):
    root_path = '../MobiFall_Dataset_v2.0' 
    subs = ['sub' + str(i) for i in range(1, 25)]
    # print(subs)
    acc_data = []
    gyro_data = []
    ori_data = []
    data_file_names = []
    classes = {'STD':0,'WAL':1,'JOG':2,'JUM':3,'STU':4,'STN':5,'SCH':6,'CSI':7,'CSO':8,'FOL':9,'FKL':10,'BSC':11,'SDL':12}
    for sub in subs: # sub
        print(sub)
        folders = os.listdir(os.path.join(root_path, sub))
        for folder in folders: #FALLS
            data_folder = os.listdir(os.path.join(root_path, sub, folder))
            for data in data_folder: #BSC
                # print(data)

                data_names = os.listdir(os.path.join(root_path,sub,folder, data))
                for name in data_names:
                    # df = np.loadtxt(os.path.join(root_path,sub,folder, data, name), delimiter=",", skiprows=16)
                    df = pd.read_csv(os.path.join(root_path,sub,folder, data, name), delimiter=",", skiprows=16, names=['timestamp', 'x', 'y', 'z'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    df =df.drop_duplicates()
                    df = df.resample(str(resampling_rate) + 'ms').bfill()
                    df['activity'] = classes[data]

                    if 'acc' in name:
                        acc_data.append(df)
                    elif 'gyro' in name:
                        gyro_data.append(df)
                    elif 'ori' in name:
                        ori_data.append(df)

    for i in range(len(acc_data)):
        if gyro_data[i].shape[0] > acc_data[i].shape[0]:
            gyro_data[i] = gyro_data[i][:acc_data[i].shape[0]]
        else:
            acc_data[i] = acc_data[i][:gyro_data[i].shape[0]]
    acc_data = np.vstack(acc_data)
    gyro_data = np.vstack(gyro_data)
    ori_data = np.vstack(ori_data)
    return acc_data, gyro_data, ori_data

def denoise_data(data):
    b, a = butter(4, 0.04, 'low', analog=False)
    for i in range(3):
        data[:, i] = lfilter(b, a, data[:, i])
    return data

def save_data(data, name):
    df = pd.DataFrame(data, columns = ['x', 'y', 'z', 'activity'])
    df.to_csv(name + '.csv', index=False)

acc_data, gyro_data, ori_data = load_data()
acc_data = denoise_data(acc_data)
gyro_data = denoise_data(gyro_data)
save_data(acc_data, 'acc_data')
save_data(gyro_data, 'gyro_data')
