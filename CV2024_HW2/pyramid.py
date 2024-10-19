import cv2
import os
import glob
import numpy as np


def gaussian_kernel(size, sigma=1):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = (1 / (2.0 * np.pi * sigma**2)) * \
        np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # zero_padding
    padded_image = np.pad(image, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant', constant_values=0)

    output_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output_image[i, j] = np.sum(region * kernel)

    return output_image


def gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)

    channels = cv2.split(image)
    blurred_channels = [convolve2d(ch, kernel) for ch in channels]
    blurred_image = cv2.merge(blurred_channels)

    return blurred_image


def downsample(image, factor):
    if factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1")

    channels = cv2.split(image)
    downsampled_channels = [ch[::factor, ::factor] for ch in channels]
    downsampled_image = cv2.merge(downsampled_channels)

    return downsampled_image


def gaussian_pyramid(image, num_layers):
    pyramid = [image]
    for i in range(num_layers - 1):
        image = gaussian_blur(image, kernel_size=5, sigma=1)
        image = downsample(image, factor=2)
        pyramid.append(image)
    return pyramid


def save_pyramid(pyramid, output_dir, base_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, layer in enumerate(pyramid):
        layer_filename = f"{base_name}_layer_{i}.jpg"
        layer_path = os.path.join(output_dir, layer_filename)
        cv2.imwrite(layer_path, layer)
        print(f"Saved: {layer_path}")


def process_images_in_directory(input_dir, num_layers):
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + \
        glob.glob(os.path.join(input_dir, '*.bmp'))

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot open image: {image_path}")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        pyramid = gaussian_pyramid(image, num_layers)

        output_dir = f"output/{base_name}"

        save_pyramid(pyramid, output_dir, base_name)


input_dir = 'data/task1and2_hybrid_pyramid'

process_images_in_directory(input_dir, 7)
