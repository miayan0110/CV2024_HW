import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from align import image_pyramid_align

def split_channels(image):
    h = image.shape[0] // 3
    blue_ch = image[:h]
    green_ch = image[h:h*2]
    red_ch = image[2*h:h*3]
    return blue_ch, green_ch, red_ch


def crop_image(image, crop_fraction=0.1):
    h, w = image.shape
    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)
    return image[crop_h:h-crop_h, crop_w:w-crop_w]

def combine_channels(blue_ch, green_ch, red_ch):
    return np.dstack((red_ch, green_ch, blue_ch))
    # return np.dstack((blue_ch, green_ch, red_ch))

def main(data_path, save_path):
    img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    blue_ch, green_ch, red_ch = split_channels(img)
    # Crop
    blue_ch = crop_image(blue_ch)
    green_ch = crop_image(green_ch)
    red_ch = crop_image(red_ch)

    # Align green and red channels to the blue channel using an image pyramid
    for _ in range(2):
        green_ch = image_pyramid_align(blue_ch, green_ch, levels=4, window_size=30)
        red_ch = image_pyramid_align(blue_ch, red_ch, levels=4, window_size=30)

    # Combine the aligned channels into an RGB image
    rgb_img = combine_channels(blue_ch, green_ch, red_ch)

    plt.imshow(rgb_img)
    plt.savefig(save_path)
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    img_list = os.listdir(r'./task3_colorizing/custom')
    
    for img_name in img_list:
        print(f'Processing : {img_name}')
        save_img = img_name.split('.')[0]
        save_path = rf'./out/{save_img}.jpg'
        main(data_path=rf'./task3_colorizing/custom/{img_name}', 
             save_path=save_path)
