# Importing the packages
from math import log10, sqrt
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.util import random_noise
from imageio import imread

################################################### Part 1 ###################################################
''' To perform basic image Analysis, adding some random noise to genrate nosiy image 
    and check basic info of the gven input. '''
################################################### ###### ###################################################

# Loading the image
img = imread('/Users/aneruthmohanasundaram/Documents/GitHub/Project_Alina/Data/Test3.jpg')

# A function to add random noise to our image
def add_noise_to_image(image):
  ''' Considering only the gaussian noise filtering but we can choose different noise by changing the mode'''
  noise_image = random_noise(image,mode='gaussian',seed=None, clip=True)
  return noise_image

# A function to check the Peak Signal Ratio
def psnr(orgImg, nosiyImage): 
	mse = np.mean((orgImg - nosiyImage) ** 2) 
	if(mse == 0):return 100
	maximum_pixel_can_be_used = 255.0 # This number can be cahnged according to our wish
	ratio = 20 * log10(maximum_pixel_can_be_used / sqrt(mse)) 
	return ratio

def rgbToBlack(image):
  image_copy = image
  height, width, channels = image.shape
  for i in range(height):
    for j in range(width):
        image_copy[i,j] = 0.3 * img[i,j][0] + 0.59 * img[i,j][1] +  0.11 * img[i,j][2]
  return image_copy

# To perform image analysis
print(f'Type of the given image is {type(img)}'+ '\n')
print(f'Size of given image is {img.shape}' + '\n')
print(f'Height of given image is {img.shape[0]}'+ '\n')
print(f'Width of given image is {img.shape[1]}'+ '\n')
print(f'Diemension of the given image is {img.ndim}' + '\n')
print(f'PSNR value is: {psnr(img, add_noise_to_image(img))} dB')

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

################################################### Part 2 ###################################################
''' A bilateral filter is a non-linear, edge-preserving, noise-reducing image smoothing filter. It replaces 
each pixel's intensity with a weighted average of intensity values from surrounding pixels. A Gaussian 
distribution can be used to calculate this weight. '''
################################################### ###### ###################################################