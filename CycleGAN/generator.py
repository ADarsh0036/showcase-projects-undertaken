import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode = "reflect"), nn.InstanceNorm2d(out_channels), nn.ReLU())
        
    def forward(self, x):
        
        return self.conv(x)
    
class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvTBlock, self).__init__()
        
        self.convT = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, **kwargs), nn.InstanceNorm2d(out_channels), nn.ReLU())
    
    def forward(self, x):
        
        return self.convT(x)
    

class ResBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ResBlk, self).__init__()
        
        self.resblk = nn.Sequential(ConvBlock(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1), ConvBlock(out_channels, out_channels, stride = 1, kernel_size = 3, padding = 1))
        
    def forward(self, x):
        
        return x + self.resblk(x)
    
class Generator(nn.Module):
    
    def __init__(self, in_channels, out_channels = 3, num_features = 64, num_residuals = 9):
        super(Generator, self).__init__()
        
        self.convinit = nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect"), nn.InstanceNorm2d(num_features), nn.ReLU())
        
        self.down128 = ConvBlock(num_features, num_features * 2, stride = 2, kernel_size = 3, padding = 1)
        self.down256 = ConvBlock(num_features*2, num_features * 4, stride = 2, kernel_size = 3, padding = 1)
        
        self.resblk = nn.Sequential(*[ResBlk(num_features * 4, num_features * 4) for _ in range(num_residuals)])
        
        self.up128 = ConvTBlock(num_features * 4, num_features * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.up64 = ConvTBlock(num_features * 2, num_features, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        
        self.convfinal = nn.Conv2d(num_features, in_channels, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect")
        
    
    def forward(self, x):
        
        x = self.convinit(x)
        x = self.down256(self.down128(x))
        
        x = self.resblk(x)
        
        x = self.up64(self.up128(x))
        
        x = self.convfinal(x)
        
        return x
    

img_channels = 3
img_size = 256
x = torch.randn((2, img_channels, img_size, img_size))
gen = Generator(img_channels, 9)
print(gen(x).shape)
    