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

        img = Image.open(img_path)
        img_trans = self.transform(img, self.phase).to(self.device)
        label = torch.tensor(loadLabel(lab_path, ROW, COL), dtype=torch.float32, device=self.device)
        label = torch.reshape(label, (-1,))

        return img_trans, label


dir = make_data_path_list("./Data", "train")
myDataset = MyDataset(dir, ImageTransform(224))