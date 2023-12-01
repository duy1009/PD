# train.py
IMG_SIZE_TRAIN = (144, 256)
BATCH_SIZE = 16
EPOCHS = 500
SAVE_PATH = "./weight.pth"
SAVE_EACH_EPOCH = True
PRE_MODEL = ""
DATA_ROOT = r"E:\DATN\Dataset\v6"
ROW = 5
COL = 9

# main.py
WEIGHT_LOAD = "./weightv5_2.pt"
ARM = [-20, 100.5, -5.75]
IMG_SIZE = (144, 256)
UAV_VELOCITY = 2