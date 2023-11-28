import airsim
from model import UAM8
from config import *
from utils import *
from _process_dataset import ImageTransform
import torch
from PIL import Image
import time 
import glob
import matplotlib.pyplot as plt
from utils import loadLabel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ROW, COL = 5, 9
THRES = 0.8
w, h = (1, 1)
ROOT = r"E:\DATN\Dataset\indoor_images"
def replaceArea(img, index, value, img_ori):
    img = img.copy()
    # w, h = img.shape[1]//COL, img.shape[0]//ROW
    x, y = index[0]*w, index[1]*h
    if value == 0:
        img[y:y+h, x:x+w] = img_ori[y:y+h, x:x+w]
    if value == 1:
        red = np.zeros((h,w,3), dtype=np.uint8)
        red[:,:] = (0, 0, 255)
        img[y:y+h, x:x+w] = cv2.addWeighted(img_ori[y:y+h, x:x+w], 0.6, red, 0.4, 0)
    return img
def updateImg(img_ori, label):
    img = img_ori.copy()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img = replaceArea(img, (j, i), label[i, j], img_ori)
    return img

datatest = glob.glob(os.path.join(ROOT, "images", "**"))
labels = glob.glob(os.path.join(ROOT, "labels", "**"))
net = load_model(UAM8(), r"E:\DATN\Code\PD\weightv5.pt")
transform = ImageTransform(IMG_SIZE)
cnt = 0

out_true = []
out_pred = []
while True:
    imgcv2 = cv2.imread(datatest[cnt])
    img = Image.fromarray(imgcv2)
    img = torch.unsqueeze(transform(img, "train"), dim=0)
    w, h = imgcv2.shape[1]//COL, imgcv2.shape[0]//ROW
    # predict
    out = net(img)[0]
    out[out<THRES] = 0
    out[out>=THRES] = 1
    out = torch.reshape(out, (ROW, COL))
    out = torch.tensor(out, dtype=torch.uint8)
    # print("Output:", out)

    out_pred += out.flatten().tolist()
    namef,_ = getNameFile(datatest[cnt])
    out_true += loadLabel(os.path.join(ROOT, "labels", namef+".txt"), ROW, COL).flatten().tolist()
    imgshow = updateImg(imgcv2, out)
    cv2.imshow("im", imgshow)
    key = cv2.waitKey(1)
    cnt+=1
    if key == ord("q") or cnt >= len(datatest):
        break
    # elif key == ord("d"):
    

    
cm = confusion_matrix(out_true, out_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
