
from config import *
from _process_dataset import make_data_path_list, MyDataset, ImageTransform
from math import pi
import matplotlib.pyplot as plot
import numpy as np
from utils import angle2D


data_root = r"D:\AirSim\aumi-set\Data\train"  # Your dataset path
idle_thres = 0.5 # minimum velocity value that is considered to be moving


list_train = make_data_path_list(root = data_root, phase="")
print(f"[DATA]:", len(list_train["envs"]),"items found!")
dataset = MyDataset(list_train, transform=ImageTransform(IMG_SIZE_TRAIN), phase="train")

data_ang = [0]*360
idle = 0

for inp, lab in dataset:
    img, env = inp
    # if not moving
    if np.linalg.norm(lab[:2]) < idle_thres or np.linalg.norm(env[0,:2]) < idle_thres:
        idle+=1
    else:
        ang = angle2D(env[0,:2] - lab[:2])*180/pi - 270 # calculate angle yaw (-180, 180)
        data_ang[int(ang)] += 1
print("Number of non-moving:", idle)

fig, ax = plot.subplots()
ax.bar(range(-180, 180), data_ang)
ax.set_ylabel('Count')

plot.show()
# plot.plot(data_ang)