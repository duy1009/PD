import airsim
from model import UAM8
from config import *
from utils import *
from _process_dataset import ImageTransform
import torch
from PIL import Image
import time 
import math

trans = ImageTransform(IMG_SIZE_TRAIN)
img = Image.open(r"E:/DATN/Dataset/v6/train/images/boxung_idle_6.png").convert('RGB')
print(img)
img_trans = trans(img,'train')

print(img_trans.shape)