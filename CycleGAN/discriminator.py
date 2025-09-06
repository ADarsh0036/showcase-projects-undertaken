import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 2):
        super(CNNBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,1,bias=True,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self,in_channels = 3,features = 64):
        super(Discriminator,self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,out_channels=features,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.c128 = CNNBlock(features, features * 2,stride= 2)
        self.c256 = CNNBlock(features *2, features * 4, stride = 2)
        self.c512 = CNNBlock(features *4, features * 8, stride = 1)
        self.cfinal = nn.Conv2d(features * 8,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect')
    
    def forward(self,x):
        x = self.initial(x)
        # x = self.c256(self.c128(x))
        x = self.c512(self.c256(self.c128(x)))
        x = self.cfinal(x)
        return torch.sigmoid(x)
    
    

x = torch.randn((5, 3, 256, 256))
model = Discriminator(in_channels=3)
preds = model(x)
print(preds.shape)
