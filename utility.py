import os
import torch
from PIL import Image
from skimage import io
import numpy as np
import cv2

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def evaluate_accuracy(data_loader, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  #make sure network is in evaluation mode

    #init
    # acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
    acc_sum = 0
    n = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        X = X.float()
        y = y.float()

        with torch.no_grad():
            # y = y.long()
            # acc_sum += torch.sum((torch.argmax(net(X)[1], dim=1) == y))
            
            acc_sum += torch.sum(net(X)[1] == y)

            n += y.shape[0] #increases with the number of samples in the batch

    return acc_sum/n

def save_rgb_image(rgb_tensor, epoch, batch_idx, save_dir='output_images'):
    """
    Save a batch of RGB images to disk.
    :param rgb_tensor: torch.Tensor, shape (B, C, H, W)
    :param epoch: int, epoch number
    :param batch_idx: int, batch index
    :param save_dir: str, directory to save images
    :return: str, path to saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, f'batch_{epoch}_img{batch_idx}.png')
    
    # (224, 224, 3)
    # Is in the range 0-1
    rgb_tensor *= 255.0
    rgb_tensor = rgb_tensor.astype(np.uint8)
    
    # cv2.imwrite(image_path, rgb_tensor)
    
    io.imsave(image_path, rgb_tensor)
    