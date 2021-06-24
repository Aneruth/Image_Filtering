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
img = imread('/Users/aneruthmohanasundaram/Documents/GitHub/Project_Alina/Data/Test1.PNG')

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

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,8))

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
plt.show()

# Intensity of an image
plt.xlabel('Intensity Value')
plt.ylabel('Count')
plt.title('Intensity histogram')
plt.hist(grayImg.ravel(), bins = 250, cumulative = True)
plt.show()

################################################### Part 2 ###################################################
''' A bilateral filter is a non-linear, edge-preserving, noise-reducing image smoothing filter. It replaces 
each pixel's intensity with a weighted average of intensity values from surrounding pixels. A Gaussian 
distribution can be used to calculate this weight. '''
################################################### ###### ###################################################

I = np.lib.pad(noisy_img, 1, 'mean') # Calculating the mean value of each pixel
bilateral_iamge = np.copy(noisy_img) # Making a copy data of image

def bilateral_filter(i,j,d,I,sigma_d,sigma_r):
    arr,sum_num,sum_den=[],0,0

    # Asigning the distance value for each neighbourhood pixel
    def distance(i, j):
      return np.absolute(i-j) # returns the absolute position of each pixel
    
    ''' assigining the kernel size for instance considering the kernel size to be 5X5 pixels. '''
    for k in range(i-floor(d/2),i+ceil(d/2)):
        for l in range(j-floor(d/2),j+ceil(d/2)):
            term = (((i-k)**2)+(j-l)**2)/(sigma_d**2*2) + (distance(I[i,j],I[k,l]))/(sigma_r**2*2)
            w = exp(-term) # Assigning the weights
            arr.append(w)
            sum_num += (I[k,l]*w)
            sum_den += w      
    return sum_num/sum_den

plt.imsave('/Users/aneruthmohanasundaram/Documents/GitHub/Project_Alina/Data/alina_output.png',bilateral_iamge,cmap='gray')

print(f'Type of the given bilateral image is {type(bilateral_iamge)}')
print(f'Size of given bilateral image is {bilateral_iamge.shape}')
print(f'Height of given bilateral image is {bilateral_iamge.shape[0]}')
print(f'Width of given bilateral image is {bilateral_iamge.shape[1]}')
print(f'Diemension of the given bilateral image is {bilateral_iamge.ndim}')
print(f'PSNR value for bilateral image is: {psnr(noisy_img,bilateral_iamge)} dB')

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
axes[2].imshow(bilateral_iamge,cmap='gray')
plt.show()