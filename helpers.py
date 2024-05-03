from PIL import Image
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def rescale(input_image_path, height=224, width=224):
    with Image.open(input_image_path) as img:
        return img.resize((width, height))

def resizeEntireDataset(dataset_directory, output_directory='rescaled'):
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(project_root, output_directory)
    total_images_processed = 0  # to print progress
    PRINT_EVERY = 10 # to print progress
    #walk through all directories
    for root, _, files in os.walk(dataset_directory):
        num_files = len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
        images_processed = 0  # Counter for images processed in the current directory

        for file in files:
            if file.lower().endswith(('.png')):
                full_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, dataset_directory)
                new_directory = os.path.join(output_base, rel_path)
                os.makedirs(new_directory, exist_ok=True)
                
                new_file_path = os.path.join(new_directory, file)
                
                try:
                    rescaled_image = rescale(full_file_path)
                    rescaled_image.save(new_file_path)
                    images_processed += 1
                    total_images_processed += 1
                except Exception as e:
                    print(f"Could not rescale {full_file_path}: {e}")
                
                if images_processed % PRINT_EVERY == 0:
                    print(f"Resized: {images_processed} / {num_files} images in folder {root}")

        if images_processed % PRINT_EVERY != 0:
            print(f"Resized: {images_processed} / {num_files} images in folder {root}")

    print(f"Resizing done. Resized {total_images_processed} images into folder '{output_directory}'.")


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


def calculate_loss(predicted, truth, output_size):
    """
    Calculates the mean square error loss between the predicted a*b* components from the colorization network
    and the scaled truth a*b* components.

    Parameters:
        predicted (torch.Tensor): The predicted a*b* components output from the colorization network.
        truth (torch.Tensor): The ground truth a*b* components of the target image.
        output_size (tuple): The desired output size as (height, width) to which the truth tensor should be resized.

    Returns:
        torch.Tensor: The mean square error loss.
    """
    #resize truth to match size of predicted
    #use align_corners=False to avoid artifacts in the rescaling process
    truth_resized = F.interpolate(truth.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)

    mse_loss = torch.mean((predicted - truth_resized) ** 2)
    return mse_loss


def loss(predicted_colors, true_colors, predicted_class, true_class):
    colorization_loss = calculate_loss(predicted_colors, true_colors, (256, 256)) 
    classification_loss = torch.nn.CrossEntropyLoss(predicted_class, true_class)
    # Loss should be colorization - (alpha*classification)


# Example usage
resizeEntireDataset('data')
target = convert_to_lab_and_normalize('rescaled/train/Cinema/fujifilm_x_t4_43.png')
predicted = convert_to_lab_and_normalize('rescaled/train/Cinema/fujifilm_x_t4_43.png')
loss = calculate_loss(predicted, target, (224, 224)) #loss should be zero since predicted and target are the same
print("Loss:", loss)


