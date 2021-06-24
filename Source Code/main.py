# Importing the packages
from math import *
import numpy as np,cv2
from skimage import io
from matplotlib import pyplot as plt
from skimage.util import random_noise
from imageio import imread

################################################### Part 1 ###################################################
''' To perform basic image Analysis, adding some random noise to genrate nosiy image 
    and check basic info of the gven input. '''
################################################### ###### ###################################################

# Loading the image
img = imread('image path')

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
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# To perform image analysis
print(f'Type of the given image is {type(img)}'+ '\n')
print(f'Size of given image is {img.shape}' + '\n')
print(f'Height of given image is {img.shape[0]}'+ '\n')
print(f'Width of given image is {img.shape[1]}'+ '\n')
print(f'Diemension of the given image is {img.ndim}' + '\n')
print(f'PSNR value is: {psnr(img, add_noise_to_image(img))} dB')

grayImg = rgbToBlack(img)
noisy_img = add_noise_to_image(grayImg)

fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(15,8))

# to plot the image
axes[0].set_xlabel('Width')
axes[0].set_ylabel('Height')
axes[0].set_title('Orginal Image')
axes[0].imshow(grayImg,cmap='gray')

# hist function is used to plot the histogram of an image.
axes[1].set_xlabel("Value")
axes[1].set_ylabel("pixels Frequency")
axes[1].set_title(" Histogram for given Image")
axes[1].hist(img[:,:,0]) # Using the image slicing to convert the 3D image to 2D image for the sake of plotting

# to plot the noisy image
axes[2].set_xlabel('Width')
axes[2].set_ylabel('Height')
axes[2].set_title('Noisy Image')
axes[2].imshow(add_noise_to_image(grayImg),cmap='gray')

# Intensity of an image
axes[3].set_xlabel('Intensity Value')
axes[3].set_ylabel('Count')
axes[3].set_title('Intensity histogram')
axes[3].hist(grayImg.ravel(), bins = 250, cumulative = True)

plt.show()

################################################### Part 2 ###################################################
''' A bilateral filter is a non-linear, edge-preserving, noise-reducing image smoothing filter. It replaces 
each pixel's intensity with a weighted average of intensity values from surrounding pixels. A Gaussian 
distribution can be used to calculate this weight. '''
################################################### ###### ###################################################

I = noisy_img
data = I
I = np.lib.pad(I, 1, 'mean')
I_new = np.copy(data)

def bilateral_filter(height,width,d,I,sigma_d,sigma_r):
    arr=[]
    sum_num=0
    sum_den=0
    
    # Asigning the distance value for each neighbourhood pixel
    def distance(height, width):
      return np.absolute(height-width) # returns the absolute position of each pixel
    
    ''' assigining the kernel size for instance considering the kernel size to be 5X5 pixels. '''
    for k in range(height-floor(d/2),height+ceil(d/2)):
        for l in range(width-floor(d/2),width+ceil(d/2)):
            term = (((height-k)**2)+(width-l)**2)/(sigma_d**2*2) + (distance(I[height,width],I[k,l]))/(sigma_r**2*2)
            w = exp(-term) # Assigning the weights
            arr.append(w)
            sum_num += I[k,l]*w
            sum_den += w      
    return sum_num/sum_den

def show_bilateral_image(image,sigma_d,sigma_r):
  height,width = image.shape
  for i in range(1,height):
    for j in range(1,width):
      # Considering the sigma_d,sigma_r as same value and radius as default value 10.
      I_new[i-1,j-1] = bilateral_filter(i-1,j-1,10,I,sigma_d,sigma_r)
  return I_new

bilateral_image = show_bilateral_image(noisy_img,77,77)

print(f'Type of the given bilateral image is {type(bilateral_image)}')
print(f'Size of given bilateral image is {bilateral_image.shape}')
print(f'Height of given bilateral image is {bilateral_image.shape[0]}')
print(f'Width of given bilateral image is {bilateral_image.shape[1]}')
print(f'Diemension of the given bilateral image is {bilateral_image.ndim}')
print(f'PSNR value for bilateral image is: {psnr(noisy_img,bilateral_image)} dB')

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,8))

# to plot the image
axes[0].set_xlabel('Width')
axes[0].set_ylabel('Height')
axes[0].set_title('Orginal Image')
axes[0].imshow(grayImg,cmap='gray')

# to plot the noisy image
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")
axes[1].set_title("Noisy Image")
axes[1].imshow(noisy_img,cmap='gray') 

# to plot the Bilateral image
axes[2].set_xlabel('Width')
axes[2].set_ylabel('Height')
axes[2].set_title('Bilateral Image')
axes[2].imshow(bilateral_image,cmap='gray')
plt.show()