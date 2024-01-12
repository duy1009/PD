from _process_dataset import DatasetLSTM, ImageTransform, make_data_path_list_lstm
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.lstm_model import UAM8LSTM
import torch
from torch.nn import MSELoss
from config import *
from utils import accuracy, load_model
import os

if os.path.exists(PRE_MODEL):
    net = load_model(UAM8LSTM(), PRE_MODEL)
else:
    net = UAM8LSTM()

# Create Dataloader
resize = IMG_SIZE_TRAIN
batch_size = BATCH_SIZE
data_root = DATA_ROOT
list_train = make_data_path_list_lstm(root = DATA_ROOT, phase="train", sequence_length= SEQ_LEN)
list_val = make_data_path_list_lstm(root = DATA_ROOT,phase="val", sequence_length= SEQ_LEN)

train_dataset = DatasetLSTM(list_train, transform=ImageTransform(resize), phase="train")
val_dataset = DatasetLSTM(list_val, transform=ImageTransform(resize), phase="val")

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# loss function
criterior = MSELoss()

# Define classes to learn
for name, param in net.named_parameters():
    param.requires_grad = True  # Always update

# Optimizer
# optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)


def train(net, dataloader_dict, criterior, optimizer, epochs, save_path, save_each_epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE]: {device}")
    net.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        # Train
        epoch_loss = 0
        acc = 0
        for img, label in tqdm(dataloader_dict["train"]):
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # img2 = ImageTransform.toCv2Img(img[0])
                # cv2.imshow("a", img2)
                # cv2.waitKey(0)
                out = net(img)
                loss = criterior(out, label)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()*img.size(0)
                acc+= accuracy(out, label, 0.3)*batch_size


        loss_avg = epoch_loss/len(dataloader_dict["train"].dataset)
        acc_avg = acc/len(dataloader_dict["train"].dataset)
        print(f"Train: -Loss: {loss_avg}, -acc: {acc_avg}")

        # # Val
        epoch_loss = 0
        acc = 0
        for img, label in dataloader_dict["val"]:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = net(img)
                loss = criterior(out, label)
                epoch_loss+=loss.item()*img.size(0)
                acc+= accuracy(out, label, 0.3)*batch_size

        loss_avg = epoch_loss/len(dataloader_dict["val"].dataset)
        acc_avg = acc/len(dataloader_dict["val"].dataset)
        print(f"Valid: -Loss: {loss_avg}, -acc: {acc_avg}")

        if save_each_epoch:
            torch.save(net.state_dict(), save_path)
    if not save_each_epoch:
        torch.save(net.state_dict(), save_path)



print("Parameters:", sum(p.numel() for p in net.parameters()))
train(net, dataloader_dict, criterior, optimizer, EPOCHS, SAVE_PATH, SAVE_EACH_EPOCH)
