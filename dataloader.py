import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab



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
        l, ground_truth_a_b = convert_to_lab_and_normalize(img_path) 
        # print("ground_truth_a_b shape: ", ground_truth_a_b.shape)
        return input_greyscale, ground_truth_a_b, l
    
    def get_luminance(self, idx):
        img_path = self.imgs[idx]
        l, _ = convert_to_lab_and_normalize(img_path) 
        return l


def convert_to_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tensor_img = torch.tensor(img)
    # adding a channel dimension: 1 x H x W
    tensor_img = tensor_img.unsqueeze(0)  #pytorch expects channel first
    return tensor_img

def convert_to_lab_and_normalize(image_path):
    """
    image_path: path to the image
    returns: luminance, a numpy array of shape (224, 224)
             ab, a tensor of shape (2, 224, 224)
    """
    img = cv2.imread(image_path)
    
    
    # print("Original R Range - Max:", np.max(img[0]), np.min(img[0]))
    # print("Original G Range - Max:", np.max(img[1]), np.min(img[1]))
    # print("Original B Range - Max:", np.max(img[2]), np.min(img[2]))


    img_lab = rgb2lab(img)
    # print("Original LUMINANCE Range - Max:", np.max(img_lab[:, :, :1]), "Min:", np.min(img_lab[:, :, :1]))
    # print("Original AB Range - Max:", np.max(img_lab[:, :, 1:]), "Min:", np.min(img_lab[:, :, 1:]))

    
    img_lab[:,:,:1] = img_lab[:,:,:1] / 100.0
    img_lab[:,:,1:] = (img_lab[:,:,1:] + 128.0) / 256.0 # shape (224, 224, 3)


    img_lab = np.transpose(img_lab, (2,0,1)) # shape (3, 224, 224)
    luminance = img_lab[0]
    ab = torch.from_numpy(img_lab[1:])
    # print("Normalized AB Range - Max:", torch.max(ab), "Min:", torch.min(ab))
    return luminance, ab

def lab_to_rgb(l, ab):
    """
    l: tensor of shape (batch_size, 1, 224, 224)
    ab: tensor of shape (batch_size, 2, 224, 224)
    returns: list of RGB images
    """
    # converted_images = []
    
    batch_size = l.shape[0]
    
    assert l.shape == (batch_size, 1, 224, 224), "Expected shape of l: (batch_size, 1, 224, 224)"
    assert ab.shape == (batch_size, 2, 224, 224), "Expected shape of ab: (batch_size, 2, 224, 224)"
  
    l = l.numpy() * 100.0
    ab = (ab.detach().numpy() * 256.0) - 128.0
    
    img_stack = np.concatenate((l, ab), axis=1)

    # l = l.transpose((0, 2, 1, 3))
    # ab = ab.transpose((0, 2, 1, 3))

    # # Now, L and AB are of shape : [batch x height x width x channel]

    # img_stack = np.dstack((l, ab))

    # img_stack = img_stack.transpose((0, 1, 3, 2))
    
    # print("First on image stack L Range - Max:", np.max(img_stack[0, :, :,0]), np.min(img_stack[0, :, :,0]))
    # print("First on image stack A Range - Max:", np.max(img_stack[0, :, :,1]), np.min(img_stack[0, :, :,1]))
    # print("First on image stack B Range - Max:", np.max(img_stack[0, :, :,2]), np.min(img_stack[0, :, :,2]))
    
    # This line is CRUCIALL
    #   - torch requires tensors to be float32
    #   - thus all above (L, ab) are float32
    #   - scikit image floats are in range -1 to 1
    #   - http://scikit-image.org/docs/dev/user_guide/data_types.html
    #   - idk what's next, but converting image to float64 somehow
    #     does the job. scikit automagically converts those values.
    img_stack = img_stack.astype(np.float64)
    converted_images = lab2rgb(img_stack, channel_axis=1)
    # print("Final R Range - Max:", np.max(converted_images[0,0]), np.min(converted_images[0,0]))
    # print("Final G Range - Max:", np.max(converted_images[0,1]), np.min(converted_images[0,1]))
    # print("Final B Range - Max:", np.max(converted_images[0,2]), np.min(converted_images[0,2]))
    
    return converted_images



# def convert_to_lab_and_normalize(image_path):
#     img = cv2.imread(image_path)
    

#     #convert image to L*a*b* color space
#     lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    
#     l, a, b = cv2.split(lab_img)
#     #normalize a and b to [0,1]
#     a = (a.astype(np.float32) - 128) / 255 + 0.5  
#     b = (b.astype(np.float32) - 128) / 255 + 0.5  
    
#     # convert to tensors
#     a = torch.tensor(a).unsqueeze(0)  # add channel dimension
#     b = torch.tensor(b).unsqueeze(0)  # add channel dimension
    
#     normalized_ab = torch.cat((a, b), dim=0)  
#     #result has the shape [2,H,W], where 
#     #   first channel is the normalized a  
#     #   second channel is the normalized b 
#     return l, normalized_ab


# def lab_to_rgb(l, ab):
#     batch_size = l.shape[0]

#     assert l.shape == (batch_size, 1, 224, 224), "Expected shape of l: (batch_size, 1, 224, 224)"
#     assert ab.shape == (batch_size, 2, 224, 224), "Expected shape of ab: (batch_size, 2, 224, 224)"
    
#     rgb_images = []
    
#     for i in range(batch_size):
#         # Convert tensors to NumPy arrays and reshape
#         l_np = l[i, 0].cpu().numpy()
#         ab_np = ab[i].permute(1, 2, 0).detach().cpu().numpy()

#         # Scale 'l' to [0, 100] and convert to uint8
#         l_np = (l_np * 100).astype(np.uint8)

#         # Convert LAB to BGR
#         lab_image = np.concatenate((l_np[:, :, np.newaxis], ab_np), axis=2)
#         bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

#         # Convert BGR to RGB
#         rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
#         rgb_images.append(rgb_image)
        
#     return np.array(rgb_images)

# Example usage
# Assuming `l` and `ab` are the input arrays
# rgb_result = lab_to_rgb(l, ab)

