import torch
import cv2
from dataloader import convert_to_lab_and_normalize, lab_to_rgb
from utility import save_rgb_image

image_path = "/Users/abhishekkuber/Desktop/TU Delft/Q4/Computer Vision/Project/cvbydl2024/small_set/train/Cinema/_DSF3769.png"

luminance, ab = convert_to_lab_and_normalize(image_path)
luminance = torch.from_numpy(luminance)
luminance = luminance.unsqueeze(0).unsqueeze(0)
ab = ab.unsqueeze(0)

rgb = lab_to_rgb(luminance, ab)



save_rgb_image(rgb,13,13)