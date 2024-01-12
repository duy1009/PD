import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockResnet8(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(ResBlockResnet8, self).__init__()
        self.batch1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)

        self.batch2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.conv_out = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0)
    def forward(self, x):
        f = self.batch1(x)
        f = self.relu(f)
        f = self.conv1(f)

        f = self.batch2(f)
        f = self.relu(f)
        f = self.conv2(f)
        
        x = self.conv_out(x)
        return x + f

class UAM8LSTM(nn.Module):
    def __init__(self):
        super(UAM8LSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(3 , 2)

        self.res1 = ResBlockResnet8(32, 32)
        self.res2 = ResBlockResnet8(32, 64)
        self.res3 = ResBlockResnet8(64, 128)

        self.lstm = nn.LSTM(5120, 45, 300)

        self.af = nn.Sigmoid()

    def forward_cnn(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return torch.flatten(x, 1)
    
    def forward(self, x_seq):
        seq_cnn = []
        for i in range(x_seq.size()[1]):
            x = x_seq[:, i, :, :]
            seq_cnn.append(self.forward_cnn(x).unsqueeze(1))
        seq_cnn = torch.cat(seq_cnn, dim=1)
        x_seq, _ = self.lstm(seq_cnn)
        x_seq = self.af(x_seq)
        return x_seq
        


model = UAM8LSTM()
params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", params)

a = torch.rand((2, 5, 3, 144, 256))
print(a.size())
out = model(a)
print(out.size())


