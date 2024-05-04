import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class FilmPicturesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.imgs = self.make_dataset()

    def make_dataset(self):
        images = []
        for img_name in os.listdir(self.root_dir):
            img_path = os.path.join(self.root_dir, img_name)
            images.append(img_path)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        
        input_greyscale = convert_to_grayscale(img_path)
        # print("input_greyscale shape: ", input_greyscale.shape)
        ground_truth_a_b = convert_to_lab_and_normalize(img_path) 
        # print("ground_truth_a_b shape: ", ground_truth_a_b.shape)
        return input_greyscale, ground_truth_a_b


def convert_to_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tensor_img = torch.tensor(img)
    # adding a channel dimension: 1 x H x W
    tensor_img = tensor_img.unsqueeze(0)  #pytorch expects channel first
    return tensor_img



def convert_to_lab_and_normalize(image_path):
    img = cv2.imread(image_path)
    #convert image to L*a*b* color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    l, a, b = cv2.split(lab_img)
    #normalize a and b to [0,1]
    a = (a.astype(np.float32) - 128) / 255 + 0.5  
    b = (b.astype(np.float32) - 128) / 255 + 0.5  
    
    # convert to tensors
    a = torch.tensor(a).unsqueeze(0)  # add channel dimension
    b = torch.tensor(b).unsqueeze(0)  # add channel dimension
    
    normalized_ab = torch.cat((a, b), dim=0)  
    #result has the shape [2,H,W], where 
    #   first channel is the normalized a  
    #   second channel is the normalized b 
    return normalized_ab


