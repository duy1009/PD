# Predict safe direction
## Install libs
```
git clone https://github.com/duy1009/PD.git
cd PD
pip install -r requirements.txt
```
## Label tool
- **Prepare**:
Go to PD_labels.py file and edit:\
`line 6`: `IMGS_ROOT = "Path to your images folder"`\
`line 7`: `IMGS_ROOT = "Folder path to save labels"`\
- **Run**:
```
python PD_labels.py
```
- **Guide**:\
Turn off Unikey and Caps lock before use\
![image](https://github.com/duy1009/PD/assets/84137684/2ec41de4-38e7-4a72-831f-8d25ec00889f)

## Training

### Not in an image sequence form 
``` Network: 
- Input: (Batch, depth, width, height)
- Output: (Batch, action_probability) #len(action_probability) = ROW*COL 
```
Go to config.py file and modify values of training:
```
IMG_SIZE_TRAIN = (144, 256)  # resize image before training
BATCH_SIZE = 16
EPOCHS = 500
SAVE_PATH = "your_path_to_save/weight.pth"
SAVE_EACH_EPOCH = True
PRE_MODEL = "your pretrain model" # set "" if not use pretrain
DATA_ROOT = r"your dataset path"
ROW = 5  # Change based on your dataset
COL = 9  # Change based on your dataset
```
Go to terminal and run:
```
cd PD
python3 train.py
```
### Image sequence form 
``` Network: 
- Input: (Batch, sequence, depth, width, height)
- Output: (Batch, sequence, action_probability) #len(action_probability) = ROW*COL 
```
Go to config.py file and modify values of training:
```
IMG_SIZE_TRAIN = (144, 256)  # resize image before training
BATCH_SIZE = 16
EPOCHS = 500
SAVE_PATH = "your_path_to_save/weight.pth"
SAVE_EACH_EPOCH = True
PRE_MODEL = "your pretrain model" # set "" if not use pretrain
DATA_ROOT = r"your dataset path"
ROW = 5  # Change based on your dataset
COL = 9  # Change based on your dataset
SEQ_LEN = 5  # Sequence length
```
Go to terminal and run:
```
cd PD
python3 train_lstm.py
```

### Training your model
1. Push your model file to the `networks` folder
2. If your model is sequence go to `train_lstm.py` else go to `train.py`
3. Import your model
```
line 4: from networks.model import UAM8
-> form networks.your_model_file import your_network
```
4. Modify UAM8() in `line 12` and `line 14` to your_network()
5. Config values in config.py
6. Train your network

## Dataset
https://drive.google.com/drive/folders/1kqsH3Euh508KX8AiFAjiLOC94k9FURxR?usp=sharing
