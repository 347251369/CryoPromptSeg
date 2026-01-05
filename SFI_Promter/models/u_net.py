import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

class Unet_encoder(nn.Module):
    # U-net from noise2noise paper 
    def __init__(self, in_channels=1, nf=48, base_width=11, top_width=3 , pretrained=None):
        super(Unet_encoder, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(in_channels, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.init_weights(pretrained)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')  # 加载权重文件
            # self 代表当前的 Unet_encoder 实例，表示要给它加载权重
            self.load_state_dict(checkpoint, strict=False)  # 加载到模型中,
            # 设置 strict=False 以如果 checkpoint 中的权重和 Unet_encoder 的参数不完全匹配
            # （比如 checkpoint 里有额外的层，或者 Unet_encoder 有新的层），不会报错，但不会加载那些不匹配的参数。
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        # downsampling
        # print(x.size())
        out_u = []
        out_u.append(x)
        p1 = self.enc1(x)
        out_u.append(p1)
        p2 = self.enc2(p1)
        out_u.append(p2)
        p3 = self.enc3(p2)
        out_u.append(p3)
        p4 = self.enc4(p3)
        out_u.append(p4)
        p5 = self.enc5(p4)
        h = self.enc6(p5)
        out_u.append(h)
        
        return tuple(out_u)
      
class Unet_decoder(nn.Module):
    def __init__(self, out_channels=1, nf=48, base_width=11, top_width=5 ,pretrained=None):
        
        super(Unet_decoder, self).__init__()
        
        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+out_channels, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, out_channels, top_width, padding=top_width//2)
                                 )
        self.init_weights(pretrained)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')  # 加载权重文件
            # self 代表当前的 Unet_encoder 实例，表示要给它加载权重
            self.load_state_dict(checkpoint, strict=False)  # 加载到模型中,
            # 设置 strict=False 以如果 checkpoint 中的权重和 Unet_encoder 的参数不完全匹配
            # （比如 checkpoint 里有额外的层，或者 Unet_encoder 有新的层），不会报错，但不会加载那些不匹配的参数。
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
    def forward(self, x):
      # upsampling
        p4 = x[4]
        h = x[5]
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h,size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        p3 = x[3]
        n = p3.size(2)
        m = p3.size(3)  
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        p2 = x[2]
        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        p1 = x[1]
        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        img = x[0]
        n = img.size(2)
        m = img.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, img], 1)

        y = self.dec1(h)
        
        return y