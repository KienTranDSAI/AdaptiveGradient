import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
import torchvision.transforms as transforms
import ot
from utils import *
import pickle as pkl
from sklearn.metrics import pairwise_distances
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to (224, 224)
    transforms.ToTensor(),          # Convert to PyTorch tensor
])
def resize_image_rgb(image_array, target_size=(100, 100)):
    # Convert the NumPy array to a PIL image
    img = Image.fromarray(image_array.astype(np.uint8) * 255)

    # Resize the image
    resized_img = img.resize(target_size)  # Use ANTIALIAS for high-quality downsampling

    # Convert the resized image back to a NumPy array
    resized_image_array = np.array(resized_img)

    return resized_image_array
def isBoder(coord, maxSize):
    if coord[0] == 0 or coord[0] == maxSize[0]-1:
        return True
    if coord[1] == 0 or coord[1] == maxSize[1]-1:
        return True
    return False
def get_border_mask(mask):
    border_mask = mask[:,:,0]
    indexes = np.where(border_mask != 0)
    border_list = []
    for i in range(len(indexes[0])):
        prod = 1
        try:
            prod *= border_mask[indexes[0][i]-1, indexes[1][i]]
        except:
            pass
        try:
            prod *= border_mask[indexes[0][i]+1, indexes[1][i]]
        except:
            pass
        try:
            prod *= border_mask[indexes[0][i], indexes[1][i]-1]
        except:
            pass
        try:
            prod *= border_mask[indexes[0][i], indexes[1][i]+1]
        except:
            pass
        if not prod:
            border_list.append([indexes[0][i], indexes[1][i]])
    x_coords = [point[0] for point in border_list]
    y_coords = [point[1] for point in border_list]
    border_image = np.zeros(border_mask.shape)
    for point in border_list:
        border_image[point[0], point[1]] = 1
    return border_image

def create_shap_image(shap_value, standard_threshold, mask = np.zeros((224,224,3))):
    important_point = get_important_point([np.transpose(shap_value[0], (3,1,2,0))], standard_threshold)
    shap_image = np.zeros(mask.shape)[:,:,0]
    for point in important_point:
        shap_image[point[0], point[1]] = 1
def create_aig_image(grad_temp, standard_threshold, mask = np.zeros((224,224,3))):
    aig_important_points = get_important_point(get_important_val(grad_temp, get_early_epoch= True), standard_threshold)
    aig_image = np.zeros(mask.shape)[:,:,0]
    for point in aig_important_points:
        aig_image[point[0], point[1]] = 1
    return aig_image
def create_customed_image(val,standard_threshold, mask = np.zeros((224,224,3))):
    important_point = get_important_point(val, standard_threshold)
    image = np.zeros(mask.shape)[:,:,0]
    for point in important_point:
        image[point[0], point[1]] = 1
    return image
def get_cost_matrix(source, target):
    cost_matrix = pairwise_distances(source, target, metric = 'euclidean')
    return cost_matrix