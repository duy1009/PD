import os
import numpy as np
from utils import loadLabel
import torch
label = torch.reshape(torch.tensor(loadLabel("E:\DATN\Dataset\indoor_images\labels\data_9_0.txt", 3, 5), device='cpu'), (-1,))
print(label)