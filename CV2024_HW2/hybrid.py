import os
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import glob

def hybrid_image(image1, image2, D0_low=30, D0_high=30):
    # Resize images to the fit the smaller image
    if image1.shape[:2] != image2.shape[:2]:
        if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]:
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
        else:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Applying Gaussian filter on each channel
    hybrid_img = np.zeros_like(image1, dtype=np.float64)
    for channel in range(3):
        low_pass_image = gaussian_filter(image1[:, :, channel], D0_low, filter_type='low')
        high_pass_image = gaussian_filter(image2[:, :, channel], D0_high, filter_type='high')
        
        hybrid_img[:, :, channel] = low_pass_image + high_pass_image
    hybrid_img = np.clip(hybrid_img, 0, 255).astype(np.uint8)
    
    return hybrid_img

def gaussian_filter(image, D0, filter_type='low'):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Fourier transform and centered
    fft_image = fftshift(fft2(image))

    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    # Gaussian filter
    H = np.exp(-(D**2) / (2 * (D0**2))) # low-pass
    if filter_type == 'high':
        H = 1 - H                       # high-pass

    # Inverse Fourier transform
    filtered_fft = fft_image * H
    ifft_image = ifft2(ifftshift(filtered_fft))
    filtered_image = np.real(ifft_image)
    
    return filtered_image

def show_image(img, path):
    image = Image.fromarray(img)
    image.save(os.path.join('output', path))

def read_image(path):
    image = cv2.imread(path)    # reads image in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    return image

if __name__ == '__main__':
    image_paths = glob.glob('data/task1and2_hybrid_pyramid/*')  # use TA's data
    # image_paths = glob.glob('my_data/*')  # use our data
    for i in range(0, len(image_paths)-1, 2):
        image1 = read_image(image_paths[i+1])
        image2 = read_image(image_paths[i])
        hybrid_img = hybrid_image(image1, image2, D0_low=15, D0_high=15)
        show_image(hybrid_img, f'hybrid_output_{int(i/2)}.png')   # show results of TA's data
        # show_image(hybrid_img, f'hybrid_output_mydata_{int(i/2)}.png')    # show results of our data