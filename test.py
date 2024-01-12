import torch.nn as nn
import torch

deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)

img = torch.rand((1, 128, 144, 256))


a = deconv1(img)

print(a.size())