import torch
import torch.nn as nn
import torch.nn.functional as F
        
class Unet(nn.Module):
    
    def __init__(self,n_channels,n_classes):
        super(Unet,self).__init__()
        self.n_channels =  n_channels
        
        self.down0 = encoder_block(3, 16)
        self.down1 = encoder_block(16, 32)
        self.down2 = encoder_block(32, 64)
        self.down3 = encoder_block(64, 128)
        self.down4 = encoder_block(128, 256)
        
        self.center = conv_block(256, 1024)
        
        self.up4 = first_decoder_block(1024, 256)
        self.up3 = decoder_block(256, 128)
        self.up2 = decoder_block(128, 64)
        self.up1 = decoder_block(64, 32)
        self.up0 = decoder_block(32, 16)
        
        self.outconv = nn.Conv2d(16, n_classes, kernel_size=1)
                
        
    def forward(self,x):
        
        encoderpool0,encoder0 = self.down0(x)
        
        encoderpool1,encoder1 = self.down1(encoderpool0)
        encoderpool2,encoder2 = self.down2(encoderpool1)
        encoderpool3,encoder3 = self.down3(encoderpool2)
        encoderpool4,encoder4 = self.down4(encoderpool3)

        center = self.center(encoderpool4)
        
        decoder4 = self.up4(center,encoder4)
        decoder3 = self.up3(decoder4,encoder3)
        decoder2 = self.up2(decoder3,encoder2)
        decoder1 = self.up1(decoder2,encoder1)
        decoder0 = self.up0(decoder1,encoder0)

        output = self.outconv(decoder0)
        return output    


class GscnnUnet4Gates(nn.Module):
    
    def __init__(self,n_channels,n_classes):
        super(GscnnUnet4Gates,self).__init__()
        self.n_channels =  n_channels
        
        self.down0 = encoder_block(3, 16)
        self.down1 = encoder_block(16, 32)
        self.down2 = encoder_block(32, 64)
        self.down3 = encoder_block(64, 128)
        self.down4 = encoder_block(128, 256)
        
        self.center = conv_block(256, 1024)
        
        self.up4 = first_decoder_block(1024, 256)
        self.up3 = decoder_block(256, 128)
        self.up2 = decoder_block(128, 64)
        self.up1 = decoder_block(64, 32)
        self.up0 = decoder_block(32, 16)
        
        self.shape = ShapeStream(16,32,16,8,64,64,80,136)
        self.outconv = nn.Conv2d(16 + 1, 3, kernel_size=1)
        

    def forward(self,x,grad,fusemethod):
        
        encoderpool0,encoder0 = self.down0(x)
        encoderpool1,encoder1 = self.down1(encoderpool0)
        encoderpool2,encoder2 = self.down2(encoderpool1)
        encoderpool3,encoder3 = self.down3(encoderpool2)
        encoderpool4,encoder4 = self.down4(encoderpool3)

        center = self.center(encoderpool4)
        
        decoder4 = self.up4(center,encoder4)
        decoder3 = self.up3(decoder4,encoder3)
        decoder2 = self.up2(decoder3,encoder2)
        decoder1 = self.up1(decoder2,encoder1)
        decoder0 = self.up0(decoder1,encoder0)
        
        shapeoutput = self.shape(grad,encoder0,encoder1,encoder2,encoder3)
        x = torch.cat([shapeoutput, decoder0], dim=1)
        combinedout = self.outconv(x)

        return combinedout, shapeoutput     
            


class GscnnUnet5Gates(nn.Module):
    
    def __init__(self,n_channels,n_classes):
        super(GscnnUnet4Gates,self).__init__()
        self.n_channels =  n_channels
        
        self.down0 = encoder_block(3, 16)
        self.down1 = encoder_block(16, 32)
        self.down2 = encoder_block(32, 64)
        self.down3 = encoder_block(64, 128)
        self.down4 = encoder_block(128, 256)
        
        self.center = conv_block(256, 1024)
        
        self.up4 = first_decoder_block(1024, 256)
        self.up3 = decoder_block(256, 128)
        self.up2 = decoder_block(128, 64)
        self.up1 = decoder_block(64, 32)
        self.up0 = decoder_block(32, 16)
        
        self.shape = ShapeStreamV2(16,32,16,8,64,64,80,136)
        self.outconv = nn.Conv2d(17,3,kernel_size=1)
    

    def forward(self,x,grad):
        
        encoderpool0,encoder0 = self.down0(x)
        encoderpool1,encoder1 = self.down1(encoderpool0)
        encoderpool2,encoder2 = self.down2(encoderpool1)
        encoderpool3,encoder3 = self.down3(encoderpool2)
        encoderpool4,encoder4 = self.down4(encoderpool3)

        center = self.center(encoderpool4)
        
        decoder4 = self.up4(center,encoder4)
        decoder3 = self.up3(decoder4,encoder3)
        decoder2 = self.up2(decoder3,encoder2)
        decoder1 = self.up1(decoder2,encoder1)
        decoder0 = self.up0(decoder1,encoder0)
        
        shapeoutput = self.shape(grad,encoder0,encoder1,encoder2,encoder3,encoder4)
        x = torch.cat([shapeoutput, decoder0], dim=1)
        combinedout = self.outconv(x)
        
        return combinedout, shapeoutput     





    
class resnetIdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resnetIdentityBlock,self).__init__()
        print(out_channels)
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, padding = 'same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace = True)
     
    def forward(self,x):
        r = self.resnet(x)
        x = self.relu(torch.add(r,x))
        
        return x

class gatedConv(nn.Module):
    def __init__(self, in_channels):
        super(gatedConv,self).__init__()
        
        self.gatedconv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,in_channels,kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels,1,kernel_size = 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
    
    def forward(self,x):
        return self.gatedconv(x)
   
    
        
class ShapeStream(nn.Module):
    def __init__(self,v1,v2,v3,v4,g1,g2,g3,g4):
        super(ShapeStream,self).__init__()        
        
        self.conv1 = nn.Conv2d(v1, 1, 1, padding = 'same') 
        self.r1 = resnetIdentityBlock(1, g1)
        self.g1 = gatedConv(g1)
        self.relu = nn.ReLU(inplace = True)
        
        self.r2 = resnetIdentityBlock(1, v2)
        self.g2 = gatedConv(g2)
        
        self.r3 = resnetIdentityBlock(1, v3)
        self.g3 = gatedConv(g3)
        
        self.r4 = resnetIdentityBlock(1, v4)
        self.g4 = gatedConv(g4)
        
        self.conv2 = nn.Conv2d(1, 1, 1,padding='same')
        self.conv3 = nn.Conv2d(2, 1, 1,padding='same')
        self.sig = nn.Sigmoid()
        
        self.transpose1 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.transpose2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=4, padding=0)
        self.transpose3 = nn.ConvTranspose2d(128, 256, kernel_size=8, stride=8, padding=0)
        
        
    def forward(self,gradient,encoder0,encoder1,encoder2,encoder3):
        
        c1 = self.conv1(encoder0)
        c1 = self.r1(c1)
        c1 = self.g1(c1)
        c1 = self.r2(c1)
        up1 = self.transpose1(encoder1)
        c1 = torch.cat([c1,up1],dim = 1)
        c1 = self.g2(c1)
        c1 = self.r3(c1)
        up2 = self.transpose2(encoder2)
        c1 = torch.cat([c1,up2],dim = 1)
        c1 = self.g3(c1)
        c1 = self.r4(c1)
        up3 = self.transpose3(encoder3)    
        c1 = torch.cat([c1,up3],dim = 1)
        c1 = self.g4(c1)
        c2 = self.conv2(c1)
        s = self.conv2(gradient)
        s = torch.cat([c2,s],dim = 1)
        s = self.conv3(s)
        output = self.sig(s)
        return output    
    
    
class ShapeStreamV2(nn.Module):
    def __init__(self,v1,v2,v3,v4,g1,g2,g3,g4):
        super(ShapeStreamV2,self).__init__()        
        
        self.conv1 = nn.Conv2d(v1, 1, 1, padding = 'same') 
        self.r1 = resnetIdentityBlock(1, g1)
        self.g1 = gatedConv(g1)
        self.relu = nn.ReLU(inplace = True)
        
        self.r2 = resnetIdentityBlock(1, v2)
        self.g2 = gatedConv(g2)

        self.r3 = resnetIdentityBlock(1, v3)
        self.g3 = gatedConv(g3)

        self.r4 = resnetIdentityBlock(1, v4)
        self.g4 = gatedConv(g4)

        
        self.r5 = resnetIdentityBlock(1, 8)
        self.g5 = gatedConv(264)

        self.conv2 = nn.Conv2d(1, 1, 1,padding='same')
        self.conv3 = nn.Conv2d(2, 1, 1,padding='same')
        self.sig = nn.Sigmoid()


    def forward(self,gradient,encoder0,encoder1,encoder2,encoder3,encoder4):
        
        c1 = self.conv1(encoder0)
        c1 = self.r1(c1)
        c1 = self.g1(c1)
        
        c1 = self.r2(c1)
        up2 = F.interpolate(encoder1, scale_factor=2, mode='bilinear', align_corners=False)
        c1 = torch.cat([c1,up2],dim = 1)
        c1 = self.g2(c1)
        
        c1 = self.r3(c1)
        up3 = F.interpolate(encoder2, scale_factor=4, mode='bilinear', align_corners=False)
        c1 = torch.cat([c1,up3],dim = 1)
        c1 = self.g3(c1)
        
        c1 = self.r4(c1)
        up4 = F.interpolate(encoder3, scale_factor=8, mode='bilinear', align_corners=False)
        c1 = torch.cat([c1,up4],dim = 1)
        c1 = self.g4(c1)
        
        c1 = self.r5(c1)
        up5 = F.interpolate(encoder4, scale_factor=16, mode='bilinear', align_corners=False)
        c1 = torch.cat([c1,up5],dim = 1)
        c1 = self.g5(c1)
        
        c2 = self.conv2(c1)
        s = self.conv2(gradient)
        s = torch.cat([c2,s],dim = 1)
        s = self.conv3(s)
        output = self.sig(s)
        return output        
    
        
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size = (3, 3),padding = 'same'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels,kernel_size = (3, 3),padding = 'same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
    def  forward(self,x):
        return self.convblock(x)

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = conv_block(in_channels,out_channels)
        self.encoder_pool = nn.MaxPool2d(2,stride = 2)
    
    def forward(self,x):
        encode = self.encoder(x)
        return self.encoder_pool(encode), encode
    
    
class first_decoder_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels, stride = 2, kernel_size = (2, 2))
        
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.rel = nn.ReLU()
        self.conv  = conv_block(out_channels*2,out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1],dim = 1)
        x = self.bn(x) 
        x = self.rel(x)
        return self.conv(x)
    
    
class decoder_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels, stride = 2, kernel_size = (2, 2))
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.rel = nn.ReLU()
        self.conv  = conv_block(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1],dim = 1)
        x = self.bn(x) 
        x = self.rel(x)
        return self.conv(x)
