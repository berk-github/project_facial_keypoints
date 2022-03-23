## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
       
        
        # NaimishNet - Facial Key Points Detection using Deep Convolutional Neural Network
        # number of convolutional layers: 4
        # conv layer 1: 32 filters, kernelSize 4x4
        # conv layer 2: 64 filters, kernelSize 3x3
        # conv layer 3: 128 filters, kernelSize 2x2
        # conv layer 4: 256 filters, kernelSize 1x1
        
        
        # input size: 224x224
        # maxPool (also test with averagePool)
        # no padding 
        # batchnormalization 
        # dropout  
        # optimizer ? 
        # learning rate ? 
        p = 0.2   # dropout probability
        output_size = 136
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)      # 220x220x32
        self.maxPool1 = nn.MaxPool2d(2,2)     # 110x110x32
        self.dropOut1 = nn.Dropout(p)
        self.batchNorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 5)      # 106x106x64   # 108x108x64 
        self.maxPool2 = nn.MaxPool2d(2,2)      # 53x53x64     # 54x54x64 
        self.dropOut2 = nn.Dropout(p)
        self.batchNorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 5)      # 49x49x128   # 52x52x128
        self.maxPool3 = nn.MaxPool2d(2,2)       # 24x24x128   # 26x26x128
        self.dropOut3 = nn.Dropout(p)
        self.batchNorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 5)      # 20x20x256   # 24x24x256
        self.maxPool4 = nn.MaxPool2d(2,2)        # 10x10x256   # 12x12x256
        self.dropOut4 = nn.Dropout(p)
        self.batchNorm4 = nn.BatchNorm2d(256)
        
        
        self.fc1 = nn.Linear(10*10*256, 1024)
        self.dropOut_fc1 = nn.Dropout(p)
        #self.batchNorm_fc1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.dropOut_fc2 = nn.Dropout(p)
        #self.batchNorm_fc2 = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, output_size)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxPool1(F.relu(self.conv1(x)))
        #x = self.maxPool1(x)
        x = self.dropOut1(x)
        x = self.batchNorm1(x)
        
        x = self.maxPool2(F.relu(self.conv2(x)))
        #x = self.maxPool2(x)
        x = self.dropOut2(x)
        x = self.batchNorm2(x)
        
        x = self.maxPool3(F.relu(self.conv3(x)))
        #x = self.maxPool3(x)
        x = self.dropOut3(x)
        x = self.batchNorm3(x)        

        x = self.maxPool4(F.relu(self.conv4(x)))
        #x = self.maxPool4(x)
        x = self.dropOut4(x)
        x = self.batchNorm4(x)      
        
        x = x.view(x.size(0), -1)   # flatten for the fc layer
        
        x = F.relu(self.fc1(x))
        x = self.dropOut_fc1(x)
        #x = self.batchNorm_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropOut_fc2(x)
        #x = self.batchNorm_fc2(x)  
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
