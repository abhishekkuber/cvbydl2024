from PIL import Image
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json

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


def calculate_loss(predicted, truth):
    """
    Calculates the mean square error loss between the predicted a*b* components from the colorization network
    and the scaled truth a*b* components.

    Parameters:
        predicted (torch.Tensor): The predicted a*b* components of size (batch_size, 2, height, width).
        truth (torch.Tensor): The ground truth a*b* components of the target image of size (batch_size, 2, height, width).
        output_size (tuple): The desired output size as (height, width) to which the truth tensor should be resized.

    Returns:
        torch.Tensor: The mean square error loss.
    """
    #resize truth to match size of predicted
    #use align_corners=False to avoid artifacts in the rescaling process
    
    '''truth_resized = F.interpolate(truth.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)
    mse_loss = torch.mean((predicted - truth_resized) ** 2)'''

    assert predicted.shape == truth.shape
    # mse_loss = torch.mean((predicted - truth) ** 2)
    mseloss = torch.nn.MSELoss(reduction='sum')
    mse_loss = mseloss(predicted, truth)
    print("predicted",predicted[0,0,45,45].item(), "truth", truth[0,0,45,45].item(), "loss", mse_loss.item())
    return mse_loss

# TODO what do we want to do with the classification? remove everything right?
# def loss(predicted_colors, true_colors, predicted_class, true_class):
#     colorization_loss = calculate_loss(predicted_colors, true_colors, (256, 256))
#     classification_loss = torch.nn.CrossEntropyLoss(predicted_class, true_class)
#     # Loss should be colorization - (alpha*classification)
#     alpha = 0
#     return colorization_loss - alpha * classification_loss

def loss(predicted_colors, true_colors):
    colorization_loss = calculate_loss(predicted_colors, true_colors)
    return colorization_loss


# Example usage
# resizeEntireDataset('data')


