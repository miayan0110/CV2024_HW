import numpy as np
import cv2

"""error"""
def square_error(image_a, image_b):
    image_a=(image_a-image_a.mean(axis=0))/(image_a.max(axis=0)-image_a.min(axis=0))
    image_b=(image_b-image_b.mean(axis=0))/(image_b.max(axis=0)-image_b.min(axis=0))
    return np.sum((image_a - image_b) ** 2)

"""align"""
def align_channel(fix_ch, align_ch, window_size=15):
    min_error = float('inf')
    align_offset = (0, 0)

    for y_offset in range(-window_size, window_size + 1):
        for x_offset in range(-window_size, window_size + 1):
            al_ch = np.roll(align_ch, (y_offset, x_offset), axis=(0, 1)) # shift the image channel
            s_error = square_error(fix_ch, al_ch)  # calculate error

            # error
            if s_error < min_error:
                min_error = s_error
                align_offset = (y_offset, x_offset)

    align_ch = np.roll(align_ch, align_offset, axis=(0, 1))
    return align_ch

"""alignment wiht gaussian pyramid"""
def image_pyramid_align(fix_ch, align_ch, levels=4, window_size=15):
    fix_pyramid = [fix_ch]
    align_pyramid = [align_ch]

    # gaussian pyramid downsample
    for _ in range(levels - 1):
        fix_pyramid.append(cv2.pyrDown(fix_pyramid[-1])) # gaussian pyramid
        align_pyramid.append(cv2.pyrDown(align_pyramid[-1]))  # gaussian pyramid

    # Align from the smallest scale to the largest
    align_ch = align_pyramid[-1]
    for i in range(levels - 1, -1, -1):
        align_ch = align_channel(fix_pyramid[i], align_ch, window_size//(2 ** i)) # decrease window size to reduce computation
        if i > 0:
            align_ch = cv2.pyrUp(align_ch, dstsize=(fix_pyramid[i - 1].shape[1], fix_pyramid[i - 1].shape[0]))

    return align_ch