import airsim
from model import UAM8
from config import *
from utils import *
from _process_dataset import ImageTransform
import torch
from PIL import Image
import time 

client = UAVStart() # Connect to airsim and take off uav  
net = load_model(UAM8(), WEIGHT_LOAD)

transform = ImageTransform(IMG_SIZE)
while True:
    #  get current position and pitch, roll, yaw
    current_pos=client.simGetVehiclePose().position.to_numpy_array()
    pitch, roll, yaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)

    imgcv2 = getPILImg(client)
    img = Image.fromarray(imgcv2)
    img = torch.unsqueeze(transform(img, "train"), dim=0)

    # predict
    v = net(img)[0]
    print("Vector v:", v)
    next_pos = torch.tensor(current_pos) + v*1.5
    cv2.imshow("a", imgcv2)
    cv2.waitKey(1)
    # move
    client.moveToPositionAsync(next_pos[0].item(), next_pos[1].item(), next_pos[2].item(), v.pow(2).sum().sqrt().item(),
                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom)
        
    time.sleep(0.05)
            
