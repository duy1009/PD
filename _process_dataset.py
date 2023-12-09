import glob
import numpy as np
from utils import loadLabel, pil2cv2
from PIL import Image
import random
import os.path as osp
from torch import Tensor
from utils import processListPath
import torch.optim
import torch.utils.data as data
from torchvision import transforms
import os
from config import ROW, COL
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(1234) # để sinh ra số random giống nhau

np.random.seed(1234)
random.seed(1234)

# torch.backends.cudnn.deterministic = True # Dùng để giữ kết quả khi training trên GPU
# torch.backends.cudnn.benchmark = False

class ImageTransform():
    def __init__(self, resize):
        self.data_trans = {
            'train': transforms.Compose([
                # data agumentation
                transforms.Resize(resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase = 'train'):
        return self.data_trans[phase](img)
    
    def toCv2Img(img: Tensor):
        return pil2cv2(torch.Tensor.numpy(img).transpose((1, 2, 0)))



def make_data_path_list(root ="./data", phase = "train"):
    target_path_im = osp.join(root+"/"+phase +"/images/*")
    target_path_im = processListPath(glob.glob(target_path_im))

    path_im_list = []
    path_lab_list = []
    for path_im in target_path_im:

        h = "/".join(path_im.split("/")[:-3])
        e = "/".join(path_im.split("/")[-1].split(".")[:-1])
        path_lab =  f"{h}/{phase}/labels/{e}.txt"

        if os.path.exists(path_lab):
            path_im_list.append(path_im)
            path_lab_list.append(path_lab)
    return {"images":path_im_list, 
            "labels":path_lab_list}
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

def make_data_path_list_lstm(root ="./data", phase = "train", sequence_length = 5):
    target_path_im = osp.join(root+"/"+phase +"/images/*")
    target_path_im = processListPath(glob.glob(target_path_im))

    path_im_list = []
    path_lab_list = []
    # Get images and labels path
    for path_im in target_path_im:

        h = "/".join(path_im.split("/")[:-3])
        e = "/".join(path_im.split("/")[-1].split(".")[:-1])
        path_lab =  f"{h}/{phase}/labels/{e}.txt"

        if os.path.exists(path_lab):
            path_im_list.append(path_im)
            path_lab_list.append(path_lab)
    print(f"[DATA {phase}]:{len(path_im_list)} items found!")
    # split by name
    path_im_list = split_by_name(path_im_list)
    path_lab_list = split_by_name(path_lab_list)

    # split by sequence
    paths_im_seq = []
    paths_lab_seq = []
    for paths_im, paths_lab in zip(path_im_list, path_lab_list):
        paths_im_seq += split_by_sequence(paths_im, sequence_length)
        paths_lab_seq += split_by_sequence(paths_lab, sequence_length)
    
    return {"images":paths_im_seq, 
            "labels":paths_lab_seq}
    


class MyDataset(data.Dataset):
    def __init__(self, file_list, transform = None, phase = "train"):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def __len__(self):
        return len(self.file_list["images"])
    
    def __getitem__(self, index):
        img_path = self.file_list["images"][index] 
        lab_path = self.file_list["labels"][index]

        img = Image.open(img_path).convert('RGB')
        img_trans = self.transform(img, self.phase).to(self.device)
        label = torch.tensor(loadLabel(lab_path, ROW, COL), dtype=torch.float32, device=self.device)
        label = torch.reshape(label, (-1,))

        return img_trans, label
    
class DatasetLSTM(data.Dataset):
    def __init__(self, file_list, transform = None, phase = "train"):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def __len__(self):
        return len(self.file_list["images"])
    
    def loadImg(self, path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img, self.phase).to(self.device)
    def loadLabel(self, path) -> torch.Tensor:
        label = torch.tensor(loadLabel(path, ROW, COL), dtype=torch.float32, device=self.device)
        return torch.reshape(label, (-1,))
    
    def loadImgSeq(self, seq_paths):
        '''
        (batchz, seq, 3, width, height)
        '''
        seq = []
        for path in seq_paths:
            img = self.loadImg(path)
            seq.append(img.unsqueeze(0))
        return torch.cat(seq, dim=0)
    def loadLabelSeq(self, seq_paths):
        '''
        (batchz, seq, )
        '''
        seq = []
        for path in seq_paths:
            lab = self.loadLabel(path)
            seq.append(lab.unsqueeze(0))
        return torch.cat(seq, dim=0)
    def __getitem__(self, index):
        imgs_path = self.file_list["images"][index] 
        labs_path = self.file_list["labels"][index]

        imgs_trans = self.loadImgSeq(imgs_path)
        labels = self.loadLabelSeq(labs_path)
        return imgs_trans, labels

# dir = make_data_path_list_lstm(r"E:\DATN\Dataset\v6", "train")
# myDataset = DatasetLSTM(dir, ImageTransform((144, 256)))

# a, l = myDataset.__getitem__(0)

# print(l.size())


