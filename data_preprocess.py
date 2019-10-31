import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Denoise data, split data and then generate original dataset for training and testing.
'''
def load_data(resampling_rate=5):
    root_path = '../MobiFall_Dataset_v2.0' 
    subs = ['sub' + str(i) for i in range(1, 25)]
    # print(subs)
    acc_data = []
    gyro_data = []
    ori_data = []
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
    df.to_csv('dataset/' + name + '.csv', index=False)

def generate_dataset(data, window_size=400, overlap=0.5):
    X = []
    y = []
    activities = pd.unique(data['activity'])
    for act in activities:
        df_temp = data[data['activity'] == act]
        df_temp = df_temp.values
        row = 0
        total_row = df_temp.shape[0]
        while row < total_row:
            sample = df_temp[row:row+window_size, :-1]
            if sample.shape[0] == window_size:
                X.append(sample)
                y.append(int(act))
            row += int(window_size * overlap)
    X = np.array(X)
    y = np.array(y)
    return X, y

def split_dataset(df='acc_data'):
    acc_data = pd.read_csv('dataset/' + df + '.csv')
    X, y=generate_dataset(acc_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, X_test, X_val, y_val, y_train, y_test

if __name__ == "__main__":
    # acc_data, gyro_data, ori_data = load_data()
    # acc_data = denoise_data(acc_data)
    # gyro_data = denoise_data(gyro_data)
    # save_data(acc_data, 'acc_data')
    # save_data(gyro_data, 'gyro_data')   

    X_train, X_test,X_val, y_train, y_test,y_val = split_dataset()
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    np.save('dataset/acc_x_train.npy', X_train)
    np.save('dataset/acc_y_train.npy', y_train)
    np.save('dataset/acc_x_test.npy', X_test)
    np.save('dataset/acc_y_test.npy', y_test)
    np.save('dataset/acc_x_val.npy', X_train)
    np.save('dataset/acc_y_val.npy', y_train)

    # X_train, X_test,X_val, y_train, y_test,y_val = split_dataset(df='gyro_data')
    # print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    # np.save('gyro_x_train.npy', X_train)
    # np.save('gyro_y_train.npy', y_train)
    # np.save('gyro_x_test.npy', X_test)
    # np.save('gyro_y_test.npy', y_test)
    # np.save('gyro_x_val.npy', X_train)
    # np.save('gyro_y_val.npy', y_train)

