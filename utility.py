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
    # print("rgb_tensor.shape", rgb_tensor.shape)
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to numpy array and then to image
    # rgb_array = rgb_tensor.cpu().detach().numpy()
    for i, rgb_array in enumerate(rgb_tensor):
        rgb_array = (rgb_array * 255).astype('uint8')  # Assuming RGB values are in [0, 1] range
        rgb_array = np.transpose(rgb_array, (1,2,0))
        # print("Saving R Range - Max:", np.max(rgb_array[:, :, 0]), np.min(rgb_array[:, :, 0]))
        # print("Saving G Range - Max:", np.max(rgb_array[:, :, 1]), np.min(rgb_array[:, :, 1]))
        # print("Saving B Range - Max:", np.max(rgb_array[:, :, 2]), np.min(rgb_array[:, :, 2]))
        # if (i==0):
        #     print(rgb_array.shape)
        #     print("10 red pixels",rgb_array[:, :, 0][0, :10])
        #     print("10 green pixels",rgb_array[:, :, 1][0, :10])

        # Save the image
        image_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}_img_{i}_c.png')
    
        cv2.imwrite(image_path, rgb_array)
        # im = Image.fromarray(rgb_array, mode="RGB")
        # im.save(image_path)

        # io.imsave(image_path, rgb_array)
    

    

    return image_path
