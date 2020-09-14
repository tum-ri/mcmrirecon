import importlib
import torch.nn.functional as F
import torch.nn as nn
#from parts_unet import *
from . import parts_unet
importlib.reload(parts_unet)

def fft_layer(image, nchannels=24):
    
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """
    kspace=None
    for i in range(0,nchannels,2):
             
        kspace_complex = torch.fft(image[:,:,:,i:i+2], 2)
        if i == 0: 
            kspace = kspace_complex
        else:
            kspace = torch.cat([kspace, kspace_complex], dim=3)
    
    return kspace


def ifft_layer(kspace_2channel, nchannels=24):
    
    image_complex_2channel=None
    for i in range(0,nchannels,2):
             
        image_complex = torch.ifft(kspace_2channel[:,:,:,i:i+2], 2)
        if i == 0: 
            image_complex_2channel = image_complex
        else:
            image_complex_2channel = torch.cat([image_complex_2channel, image_complex], dim=3)
    
    return image_complex_2channel


            
    
    

class WNet(nn.Module):   
    
    def __init__(self, in_channels, mask, kspace_flag = True, architecture = 'ki'):
        super(WNet, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        self.kspace_flag = kspace_flag
        self.mask = mask

        self.unet_blocks = []
        self.dc_blocks = []
        self.ifft_blocks = []
        
        for c in architecture:
            
            self.dc_blocks.append(parts_unet.DC_Block(mask, kspace_flag))       
            print('dc added')
            if c == 'i': #image domain
                # do ifft (input_shape = (b_s, 24, 218, 170) )
                self.ifft_blocks.append(parts_unet.IFFT(dim=2)) #reconstruct kspace to image domain
                self.kspace_flag = False  
            else: 
                self.kspace_flag = True
            self.unet_blocks.append(UNet(in_channels, in_channels)) 
            print('unet block added')
                
        self.unet_blocks = nn.ModuleList(self.unet_blocks)
        self.dc_blocks = nn.ModuleList(self.dc_blocks)
        self.ifft_blocks = nn.ModuleList(self.ifft_blocks)
        

    def forward(self, x):    
        y = x.clone()
        
        ifft_counter = 0
        for i, c in enumerate(self.architecture):
            x = self.dc_blocks[i](x,y)
            if c == 'i':
                x = self.ifft_blocks[ifft_counter](x)
                ifft_counter +=1
            x = self.unet_blocks[i](x)
        return x
           
    

class UNet(nn.Module):
    
    def __init__(self, org_in_channels, out_channels, kshape = 3):                             
        super(UNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2)
        #shape=(b_size,channels,218,170)
        self.down1 = parts_unet.TripleConv(kshape, org_in_channels, 48)
        #shape=(b_size,channels,109,85)
        self.down2 = parts_unet.TripleConv(kshape, 48, 64)
        #shape=(b_size,channels,54,42)
        self.down3 = parts_unet.TripleConv(kshape, 64, 128)
        #shape=(b_size,channels,27,21)
        self.down4 = parts_unet.TripleConv(kshape, 128, 256) 
        
        self.up1 = parts_unet.Up(384, 128)
        self.up2 = parts_unet.Up(192, 64)
        self.up3 = parts_unet.Up(112, 48)
        
        self.outc = parts_unet.OutConv(48, out_channels)

    def forward(self, unet_input):      
        #print("First block")
        #print("Input shape: ", unet_input.shape)
        #unet_input=(b_size,channels,218,170)
        conv1 = self.down1(unet_input)
        pool1 = self.pool(conv1)
        #pool1=(b_size,channels,109,85)
        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)
        #pool2=(b_size,channels,54,42)
        conv3 = self.down3(pool2)
        pool3 = self.pool(conv3)
        #pool3=(b_size,channels,27,21)
        conv4 = self.down4(pool3)
        #conv4=(b_size,channels,27,21)
        #print("Reache din unet")
        conv5 = self.up1(conv4, conv3)
        #conv5=(b_size,channels,54,42)
        conv6 = self.up2(conv5, conv2)
        #conv6=(b_size,channels,109,86)
        conv7 = self.up3(conv6, conv1)
        #print("reached2 in unet")
        conv8 = self.outc(conv7)
        out = conv8 + unet_input
        #print("Reached 3 in unet")
        #print("Out shape Unet: ", out.shape)
        return out
    

    

    