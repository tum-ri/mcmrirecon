import torch
import torch.nn as nn
import torch.nn.functional as F


def tripleConvBLock(unet_input, kshape=(3,3)):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """
    
    
    

class TripleConv(nn.Module):
    
    def __init__(self, kshape, in_channels, out_channels, org_in_channels = None):
        super().__init__()
        
        if org_in_channels is None:
            org_in_channels = in_channels
        
        # one first_input_channel?
        self.tripleConvBlock = nn.Sequential(
            #same padding
            nn.Conv2d(org_in_channels, out_channels, kernel_size=kshape, padding=1),
            #no BatchNorm in paper mentioned
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kshape, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kshape, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.tripleConvBlock(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # paper uses nearest interpolation
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = TripleConv(kshape = 3, in_channels= in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x1, x2], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class IFFT(nn.Module):
    def __init__(self, dim = 2):
        super(IFFT, self).__init__()
        self.dim = dim

    def forward(self, x):
        # x.shape = (b_s, 24, 218, 170)
        x = torch.stack((x[:,::2,:,:], x[:,1::2,:,:]), dim=-1)
        # x.shape = (b_s, 12, 218, 170, 2)
        x = torch.ifft(x, self.dim)
        # x.shape = (b_s, 12, 218, 170, 2)
        y = (torch.zeros((x.size(0), x.size(1)*2, x.size(2), x.size(3)))).cuda()
        
        # re-interleave real and img channels
        y[:,::2,:,:] = x[:,:,:,:,0]
        y[:,1::2,:,:] = x[:,:,:,:,1]
        return y
    
    
class FFT(nn.Module):
    def __init__(self, dim = 2):
        super(FFT, self).__init__()
        self.dim = dim

    def forward(self, x):
        #print("FFT x shape: " , x.shape)
        # x.shape = (b_s, 24, 218, 170)
        x = torch.stack((x[:,::2,:,:], x[:,1::2,:,:]), dim=-1)
        #print("FFt reached1")
        # x.shape = (b_s, 12, 218, 170, 2)
        x = torch.fft(x, self.dim)
        #print("FFt reached2")
        # x.shape = (b_s, 12, 218, 170, 2)
        y = (torch.zeros((x.size(0), x.size(1)*2, x.size(2), x.size(3)))).cuda() 
        #print("FFt reached3 y shape ", y.shape)
        #print("FFt reached3 x shape ", x.shape)
        # re-interleave real and img channels
        y[:,::2,:,:] = x[:,:,:,:,0]
        y[:,1::2,:,:] = x[:,:,:,:,1]
        #print("FFt reached4")
        
        return y.cuda()
    
class SkipConnection(nn.Module):
    def __init__(self, kspace_us, mask):
        super(SkipConnection, self).__init__()
        self.kspace_us = kspace_us
        self.mask = mask
    
    def forward(self, x):
        x = torch.mul(x, self.mask) #set all pixels with the same coordinates of the originally sampled kspace-pixels to zero 
        x = torch.Add(x, self.kspace_us)
        
        return x
    
class DC_Block(nn.Module):

    def __init__(self, mask, kspace_flag):
        super(DC_Block, self).__init__()
        self.mask = mask
        self.kspace_flag = kspace_flag
        self.fft = FFT(dim=2)
        
    def forward(self, x, y):     
        if not self.kspace_flag:
            x = self.fft(x)
            
        x = torch.mul(x, self.mask) #set all pixels with the same coordinates of the originally sampled   
                                    #kspace-pixels to zero 
        x = torch.add(x, y)
        
        return x
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        
        return self.lambd(x)