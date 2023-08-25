# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import glob
import cv2 
import matplotlib.pyplot as plt

# A. Load the face image dataset 
train_path = "faces_sets/training_set/*.*.jpg"
test_path = "faces_sets/test_set/*.*.jpg"
train_images = [np.array(Image.open(file).convert("RGB")) for file in glob.glob(train_path)]
test_images = [np.array(Image.open(file).convert("RGB")) for file in glob.glob(test_path)]

# B. Convert the training images to grayscale.
train_images_gray = [np.array(Image.open(file).convert("L")) for file in glob.glob(train_path)]
test_images_gray = [np.array(Image.open(file).convert("L")) for file in glob.glob(test_path)]

# C. Perform PCA on the grayscale training images.
n_components = len(train_images_gray)
pca = PCA(n_components=n_components)
train_images_flat = np.array([img.flatten() for img in train_images_gray])
pca.fit(train_images_flat)

# D.Algorithm to complete the color of a grayscale image based on the colored image.
def restore_color(gray_img, color_img):
    
    b,g,r = cv2.split(color_img)
    np.multiply(b/b.max(), gray_img, out = b, casting ='unsafe')
    np.multiply(g/g.max(), gray_img, out = g, casting ='unsafe')
    np.multiply(r/r.max(), gray_img, out = r, casting ='unsafe')
    
    gray_colored = cv2.merge([b,g,r])
    restored_color = gray_colored
    
    return restored_color


# E. Find the most similar image from the training (using PCA coordinates), and complete the color.
for i, test_gray in enumerate(test_images_gray):
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,5))
    
    test_pca_coords = pca.transform([test_gray.flatten()])
    train_pca_coords = pca.transform([img.flatten() for img in train_images_gray])
    distances = np.sum((train_pca_coords - test_pca_coords)**2, axis=1)
    most_similar_idx = np.argmin(distances)
    completed_color_img = restore_color(test_gray, train_images[most_similar_idx])
    
    # F. Display the pictures
    test_img = Image.fromarray(test_gray)
    train_img = Image.fromarray(train_images_gray[most_similar_idx])
    colored_img = Image.fromarray(completed_color_img)
    original_img = Image.fromarray(train_images[most_similar_idx])
    
    axes[0][0].set_axis_off()
    axes[0][1].set_axis_off()
    axes[1][0].set_axis_off()
    axes[1][1].set_axis_off()
    
    axes[0][0].imshow(test_img, cmap=plt.cm.gray_r) #removing the 'cmap=plt.cm.gray_r' will show the picture in green colors, and it will be easier to see the images
    axes[0][0].set_title("gray test")
    
    axes[0][1].imshow(train_img, cmap=plt.cm.gray_r) #removing the 'cmap=plt.cm.gray_r' will show the picture in green colors, and it will be easier to see the images
    axes[0][1].set_title("gray most similar")
    
    axes[1][0].imshow(colored_img)
    axes[1][0].set_title("colored gray")
    
    axes[1][1].imshow(original_img)
    axes[1][1].set_title("colored most similar")
    
for j, test_gray2 in enumerate(test_images_gray):
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,5))
    
    train_pca_coords = pca.transform([img.flatten() for img in train_images_gray])
    average_val = np.average(train_pca_coords)
    distances2 = np.sum((train_pca_coords - average_val)**2, axis=1)
    most_similar_idx2 = np.argmin(distances2)
    completed_color_img2 = restore_color(test_gray2, train_images[most_similar_idx2])
    
    # F. Display the pictures
    test_img = Image.fromarray(test_gray2)
    train_img = Image.fromarray(train_images_gray[most_similar_idx2])
    colored_img = Image.fromarray(completed_color_img2)
    original_img = Image.fromarray(train_images[most_similar_idx2])
    
    axes[0][0].set_axis_off()
    axes[0][1].set_axis_off()
    axes[1][0].set_axis_off()
    axes[1][1].set_axis_off()
    
    axes[0][0].imshow(test_img, cmap=plt.cm.gray_r) #removing the 'cmap=plt.cm.gray_r' will show the picture in green colors, and it will be easier to see the images
    axes[0][0].set_title("gray test")
    
    axes[0][1].imshow(train_img, cmap=plt.cm.gray_r) #removing the 'cmap=plt.cm.gray_r' will show the picture in green colors, and it will be easier to see the images
    axes[0][1].set_title("gray most similar")
    
    axes[1][0].imshow(colored_img)
    axes[1][0].set_title("colored gray")
    
    axes[1][1].imshow(original_img)
    axes[1][1].set_title("colored most similar")
