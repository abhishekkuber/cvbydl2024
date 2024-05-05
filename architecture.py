import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

# NOTE : You need to rescale all the input images to 224x224!
# TODO : 1. Fusion layer

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

        self.relu = nn.ReLU()

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

        self.relu = nn.ReLU()

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
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        
        # Input image = 28x28
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 56x56 -> 56x56
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) # 56x56 -> 56x56
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1) # 112x112 -> 112x112
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1) # 112x112 -> 112x112

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest') # Used after the sigmoid layer
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    # Input is the output of the fusion layer!
    # Vector is of dimensions 256x28x28
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.upsample2(x)
        x = self.relu(self.conv4(x))
        # Output layer
        x = self.sigmoid(self.conv5(x))
        
        # Upsampling here?
        x = self.upsample3(x)

        return x
    
class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        '''
        We do this by introducing another very small
        neural network that consists of two fully-connected layers: a hidden
        layer with 256 outputs and an output layer with as many outputs as
        the number of classes in the dataset, which is N = 205 in our case.
        The input of this network is the second to last layer of the global
        features network with 512 outputs. We train this network using the
        cross-entropy loss, jointly with the MSE loss for the colorization
        network. 
        '''
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=3) # 4 classes : Cinema, ClassNeg, Velvia

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) # DO WE NEED RELU HERE?
        return x 

class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28
        self.relu = nn.ReLU()
    
    def forward(self, glf, mlf):
        glf = glf.unsqueeze(-1).unsqueeze(-1)
        glf = glf.expand(1, 256, 28, 28)

        fused = torch.cat((mlf, glf), 1)
        fused = self.relu(self.conv1(fused))
        return fused
    



class FullNetwork(nn.Module):
    def __init__(self):
        super(FullNetwork, self).__init__()
        
        self.sllf = SLLF()
        self.glf = GIF()
        self.mlf = MLF()
        self.fusionLayer = FusionLayer()
        self.colorizationNetwork = ColorizationNetwork()
        self.classificationNetwork = ClassificationNetwork()
        

    # def fusion(self, glf, mlf):
    #     glf = glf.unsqueeze(-1).unsqueeze(-1)
    #     glf = glf.expand(256, 28, 28)
    #     fused = torch.cat((mlf, glf), 0)
    #     return fused
    
    def forward(self, x):

        llf = self.sllf.forward(x)

        mlf = self.mlf.forward(llf)
        cn_output, glf = self.glf.forward(llf)
        
        # Fusion
        fused = self.fusionLayer.forward(glf, mlf)

        # Classification Network
        predicted_class = self.classificationNetwork.forward(cn_output)

        # Colorization Network
        predicted_colors = self.colorizationNetwork.forward(fused)
        print(f"PREDICTED COLORS : {predicted_colors.shape}")
        

        return predicted_class, predicted_colors
