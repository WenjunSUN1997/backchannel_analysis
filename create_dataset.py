import os
import pandas as pd

def get_data(len_block:int):
    file_list = os.listdir('data_processed')
    file_real = []
    file_fake = []
    for index in range(len(file_list)):
        if file_list[index][0] == 'T':
            if 'fake' in file_list[index]:
                file_fake.append(file_list[index])
            else:
                file_real.append(file_list[index])
    for index in range(len(file_fake)):
        if index == 0:
            data_fake = pd.read_csv('data_processed/'+file_fake[index])
        else:
            data_fake = data_fake.append(pd.read_csv('data_processed/'+file_fake[index]), ignore_index=True)

    for index in range(len(file_real)):
        if index == 0:
            data_true = pd.read_csv('data_processed/' + file_real[index])
        else:
            data_true = data_true.append(pd.read_csv('data_processed/' + file_real[index]), ignore_index=True)

    for col_name in list(data_fake):
        if str(len_block) in col_name:
            data_fake = data_fake[col_name]
            break

    for col_name in list(data_true):
        if str(len_block) in col_name:
            data_true= data_true[col_name]
            break

    data_true = pd.DataFrame({'sentence': data_true.dropna(),
                              'target': [1] * len(data_true)})
    data_fake = pd.DataFrame({'sentence': data_fake.dropna(),
                              'target': [0] * len(data_fake)})[:1282]
    return data_true, data_fake