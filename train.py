import os 
import json
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm



def create_loaders():
    '''
    Creates the test, train and validation data loaders
    
    Arguments: 
        batch_size

    Returns:


    '''
    return 0 

def create_annotations():
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(project_root)

if __name__ == '__main__':
    create_annotations()