import cv2
import glob
import os
import numpy as np
from utils import loadLabel, saveLabel
IMGS_ROOT = r"E:\DATN\Dataset\indoor_images\env1"
LABELS_ROOT = r"E:\DATN\Dataset\v3\labels"
ROW = 5
COL = 9
mouse_index = (0, 0)
w, h = (1, 1)
label = np.zeros((ROW, COL), dtype=np.uint8)
cnt = 0
value = 1
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
        # img[y:y+h, x:x+w, 0] = 0
        # img[y:y+h, x:x+w, 1] = 0
        # img[y:y+h, x:x+w, 2] = 255
    return img
def click_event(event, x, y, flags, param):
    global mouse_index, value, label, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_index = x, y
        # w, h = img_ori.shape[1]//COL, img_ori.shape[0]//ROW
        i, j = int(mouse_index[0]/w), int(mouse_index[1]/h)
        label[j, i] = value
def drawCell(img_ori, w, h):
    '''Draw cell'''
    img = img_ori.copy()
    for i in range(int(img.shape[1]/w)):
        img = cv2.line(img, (int(i*w), 0), (int(i*w), img.shape[0]), color = (0, 255, 255), thickness= 1)

    for i in range(int(img.shape[0]/h)):
        img = cv2.line(img, (0, int(i*h)), (img.shape[1], int(i*h)), color = (0, 255, 255), thickness= 1)
    return cv2.addWeighted(img_ori, 0.7, img, 0.3, 0)
def updateImg(img_ori, label):
    img = img_ori.copy()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img = replaceArea(img, (j, i), label[i, j], img_ori)
    return img


def getNameFile(path):
    file = os.path.basename(path)
    name, tail = file.split(".")
    return name, tail
# Tạo cửa sổ hiển thị hình ảnh và gắn hàm callback chuột
cv2.namedWindow('PD labels')
cv2.setMouseCallback('PD labels', click_event)
im_path = glob.glob(os.path.join(IMGS_ROOT, '*.[Jj][Pp][Gg]')) + glob.glob(os.path.join(IMGS_ROOT, '*.[Pp][Nn][Gg]'))
label_path = os.path.join(LABELS_ROOT, getNameFile(im_path[cnt])[0]+ ".txt")
label = loadLabel(label_path, ROW, COL)
while True:
    img_ori = cv2.imread(im_path[cnt])
    f_name,_ = getNameFile(im_path[cnt])
    w, h = img_ori.shape[1]//COL, img_ori.shape[0]//ROW
    # img = replaceArea(img_ori, index, value, img_ori)
    img = drawCell(img_ori, w, h)
    img = updateImg(img, label)
    cv2.imshow("PD labels", img)
    key = cv2.waitKey(100) & 0xFF
    window_status = cv2.getWindowProperty("PD labels", 0)

    if key == ord("a"):
        saveLabel(os.path.join(LABELS_ROOT, f_name + ".txt"), label)
        if cnt>0:
            cnt-=1
            label_path = os.path.join(LABELS_ROOT, getNameFile(im_path[cnt])[0]+ ".txt")
            label = loadLabel(label_path, ROW, COL)
    elif key == ord("d"):
        saveLabel(os.path.join(LABELS_ROOT, f_name + ".txt"), label)
        if cnt<len(im_path)-1:
            cnt+=1
            label_path = os.path.join(LABELS_ROOT, getNameFile(im_path[cnt])[0]+ ".txt")
            label = loadLabel(label_path, ROW, COL)
        
    elif key == ord("f"):
        value = 1
    elif key == ord("r"):
        value = 0
    elif key == ord("q") or window_status == -1:
        saveLabel(os.path.join(LABELS_ROOT, f_name + ".txt"), label)
        break


