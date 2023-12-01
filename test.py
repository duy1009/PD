import airsim
from model import UAM8
from config import *
from utils import *
from _process_dataset import ImageTransform
import torch
from PIL import Image
import time 
import math

COE = torch.tensor([[0.7, 0.725, 0.75, 0.775, 0.8, 0.775, 0.75, 0.725, 0.7], 
                [0.8, 0.825, 0.85, 0.875, 0.9, 0.875, 0.85, 0.825, 0.8],
                [0.9, 0.925, 0.95, 0.975, 1, 0.975, 0.95, 0.925, 0.9],
                [0.8, 0.825, 0.85, 0.875, 0.9, 0.875, 0.85, 0.825, 0.8],
                [0.7, 0.725, 0.75, 0.775, 0.8, 0.775, 0.75, 0.725, 0.7]])# coefficient
HFOV = math.radians(90)
VFOV = math.radians(60)
w, h = (1, 1)
def chooseDir(pred):
    ind1d = torch.argmax(pred * torch.reshape(COE, (-1, ))).item()
    return ind1d - COL*(ind1d//COL) , ind1d//COL

def getCenter(max):
    return (max-1)/2
def getAngle(x, y):
    x_trans, y_trans = getCenter(COL) - x, getCenter(ROW) - y
    pitch = VFOV*y_trans/ROW
    yaw = HFOV*x_trans/COL
    return pitch, yaw
def getPosVal(pitch, yaw, v):
    # x = v*cos(yaw)*sin(pitch)
    # y = v*sin(yaw)*sin(pitch)
    # z = v*cos(pitch)
    x = v*cos(pitch)*cos(yaw)
    y = v*cos(pitch)*sin(yaw)
    z = v*sin(pitch)
    return x, y, z
def getNextPos(current_pos, pitch, yaw, next_pitch, next_yaw, velocity):
    x, y, z = current_pos
    real_pitch, real_yaw = pitch + next_pitch, yaw + next_yaw
    x_val, y_val, z_val = getPosVal(real_pitch, real_yaw, velocity)
    return x + x_val, y + y_val, z - z_val # z < 0

def replaceArea(img, index, value, img_ori, color = (0, 0, 255)):
    img = img.copy()
    # w, h = img.shape[1]//COL, img.shape[0]//ROW
    x, y = index[0]*w, index[1]*h
    if value == 0:
        img[y:y+h, x:x+w] = img_ori[y:y+h, x:x+w]
    if value == 1:
        red = np.zeros((h,w,3), dtype=np.uint8)
        red[:,:] = color
        img[y:y+h, x:x+w] = cv2.addWeighted(img_ori[y:y+h, x:x+w], 0.6, red, 0.4, 0)
        # img[y:y+h, x:x+w, 0] = 0
        # img[y:y+h, x:x+w, 1] = 0
        # img[y:y+h, x:x+w, 2] = 255
    return img
def updateImg(img_ori, label):
    img = img_ori.copy()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img = replaceArea(img, (j, i), label[i, j], img_ori)
    return img

def getResult(pred:torch.Tensor):
    pred = pred.clone()
    pred[pred<0.5] = 0
    pred[pred>=0.5] = 1
    return pred


# pitch, yaw = 0, 0
# x_ind, y_ind = 0, 1
# pitch_pred, yaw_pred = getAngle(x_ind, y_ind)
# print(pitch_pred, yaw_pred)
# next_pos = getNextPos((0, 0, 0), pitch, yaw, pitch_pred, yaw_pred, UAV_VELOCITY*1.5)

# print(next_pos)
# yaw_deg = math.degrees(yaw + yaw_pred)

client = UAVStart() # Connect to airsim and take off uav  

client.moveToPositionAsync(0, 3, -1, UAV_VELOCITY,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                           ).join()
    
time.sleep(0.05)
            
