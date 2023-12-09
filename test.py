import torch
import torch.nn as nn
import os.path as osp
from _process_dataset import processListPath
import glob
from config import DATA_ROOT
# Khởi tạo LSTM với số chiều đầu vào là 10, số chiều đầu ra là 20, và 2 lớp LSTM
lstm = nn.LSTM(5120, 45, 3)

# Tạo dữ liệu đầu vào
input = torch.randn(8, 6, 5120)  # Kích thước batch_size x sequence_length x input_size

# Áp dụng LSTM cho dữ liệu đầu vào
output, (h_n, c_n) = lstm(input)

# In ra kích thước của đầu ra và hidden state cuối cùng
# print(output.size())  # Kích thước batch_size x sequence_length x hidden_size
# print(h_n.size())    # Kích thước num_layers x batch_size x hidden_size

def get_file_name(path):
    name = osp.basename(path).split(".")[0]
    cnt = 0
    for i in range(len(name)):
        if name[i:].isalnum():
            cnt = i
            break
    return name[:cnt]
def get_index_file(path):
    len_name = len(get_file_name(path))
    file = osp.basename(path).split(".")[0]
    num = int(file[len_name:])
    return num
def append_by_index(paths:list, path_append):
    ind_path_append = get_index_file(path_append)
    ind = len(paths)
    for i, path in enumerate(paths):
        ind_path = get_index_file(path)
        if  ind_path > ind_path_append:
            ind = i
            break
    paths.insert(ind, path_append)
    return paths

def split_by_name(paths:list):
    list_paths = []
    while len(paths) > 0:
        name_temp = get_file_name(paths[0])
        list_temp = []
        for path in paths[::-1]:
            if get_file_name(path) == name_temp:
                list_temp = append_by_index(list_temp, path)
                # list_temp.append(path)
                paths.remove(path)
        list_paths.append(list_temp)
    return list_paths

def split_by_sequence(paths, sequence):
    '''
    path = ['a_0.png', 'a_1.png', 'a_2.png',...] (length>sequence)
    '''
    paths_seq = []
    for i in range(len(paths)-sequence):
        paths_seq.append(paths[i:i+sequence])

    return paths_seq

root = DATA_ROOT
phase="train"
target_path_im = osp.join(root+"/"+phase +"/images/*")
target_path_im = processListPath(glob.glob(target_path_im))

list_f = split_by_name(target_path_im)

a = split_by_sequence(list_f[0], 5)


from _process_dataset import make_data_path_list_lstm

list = make_data_path_list_lstm(DATA_ROOT)
print(list["images"])