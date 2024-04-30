import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

# NOTE : You need to rescale all the input images to 224x224!

# Shared Low Level Features
class SLLF(nn.Module):
    def __init__(self):
        
        # Output channels for each convolution layer are 64, 128, 128, 256, 256, 512. See table 1 for the details

        # For conv1, conv3, conv5, stride is 2. Therefore, it halves the height and width of the image.
        super(SLLF, self).__init__()

        # Input image = 224x224
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1) # 224x224 -> 112x112
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 112x112 -> 112x112
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1) # 112x112 -> 56x56
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 56x56 -> 56x56
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1) # 56x56 -> 28x28
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28

        self.relu = F.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        return x
    

# Global Image Features 
class GIF(nn.Module):
    def __init__(self):
        super(GIF, self).__init__()
        
        # Input image = 28x28
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1) # 14x14 -> 7x7
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # 7x7 -> 7x7
        
        # Input to this layer is a feature map of dimension 512x7x7 = 25088
        # To pass it into the linear layers, you need to flatten the feature map
        self.fc1 = nn.Linear(in_features=25088, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)


        self.relu = F.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # print(f"Printing from the GLF forward function\nDimensions before flattening {x.shape}")

        # After all convolutions, flatten the input before passing them to the Fully Connected layers
        x = torch.flatten(x, 1)
        
        # print(f"Printing from the GLF forward function\nDimensions after flattening {x.shape}")
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Output to the classification network
        cn_output = x

        # Output to the fustion layer
        x = self.relu(self.fc3(x))

        return cn_output, x    
            

# Mid Level Features
class MLF(nn.Module):
    def __init__(self):
        super(MLF, self).__init__()

        # Input image = 28x28
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28

    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


# class FusionLayer(nn.Module):
#     def __init__(self):
#         super(FusionLayer, self).__init__()

#         '''
#         Input to the layer is : 
#         1. Output from the Mid-Level Features Network (256x28x28)
#         2. Output from the Global Features Network (vector of length 256) 

#         Uses only one Linear layer    
    
#         This can be thought of as concatenating the global features with the
#         local features at each spatial location and processing them through a
#         small one-layer network. 

#         '''

#         self.linear1()

#     def forward(self, x):
