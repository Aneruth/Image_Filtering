# Importing the packages
from math import log10, sqrt
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from imageio import imread

################################################### Part 1 ###################################################
''' To perform basic image Analysis, adding some random noise to genrate nosiy image 
    and check basic info of the gven input. '''
################################################### ###### ###################################################

# Loading the image
img = imread('/Users/aneruthmohanasundaram/Documents/GitHub/Project_Alina/Data/Test2.png')

# To perform image analysis
print(f'Type of the given image is {type(img)}'+ '\n')
print(f'Size of given image is {img.shape}' + '\n')
print(f'Height of given image is {img.shape[0]}'+ '\n')
print(f'Width of given image is {img.shape[1]}'+ '\n')
print(f'Diemension of the given image is {img.ndim}')

# A function to add random noise to our image
def add_noise_to_image(image):
  ''' Considering only the gaussian noise filtering but we can choose different noise by changing the mode'''
  noise_image = random_noise(image,mode='gaussian',seed=None, clip=True)
  return noise_image

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,8))

# to plot the image
axes[0].set_xlabel('Width')
axes[0].set_ylabel('Height')
axes[0].set_title('Orginal Image')
axes[0].imshow(img)

# hist function is used to plot the histogram of an image.
axes[1].set_xlabel("Value")
axes[1].set_ylabel("pixels Frequency")
axes[1].set_title(" Histogram for given Image")
axes[1].hist(img[:,:,0]) # Using the image slicing to convert the 3D image to 2D image for the sake of plotting

# to plot the noisy image
axes[2].set_xlabel('Width')
axes[2].set_ylabel('Height')
axes[2].set_title('Noisy Image')
axes[2].imshow(add_noise_to_image(img))
plt.show()