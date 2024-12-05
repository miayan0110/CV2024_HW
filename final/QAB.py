import glob
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt



def histogram_polarization(image, T_b, T_w, savedir):
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'./output/{savedir}/Ig_y.jpg', luminance)

    # histogram polarization
    L = 256
    hist, bins = np.histogram(luminance.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # normalize CDF to the range [0, 1]
    
    mapped_values = (L - T_w + T_b) * cdf_normalized
    polarized_luminance = np.interp(luminance.flatten(), bins[:-1], mapped_values)
    
    # Adjust pixels in the high contrast range
    polarized_luminance[polarized_luminance >= T_b] += T_w - T_b
    polarized_luminance = np.clip(polarized_luminance, 0, 255)
    polarized_luminance = polarized_luminance.reshape(luminance.shape).astype(np.uint8)

    return polarized_luminance


if __name__ == '__main__':
    img_path = glob.glob('./data/image/*')
    qrc_path = glob.glob('./data/qrcode/*')

    for imgp, qrcp in zip(img_path, qrc_path):
        basename = os.path.basename(imgp).split('.')[0]
        os.makedirs(f"./output/{basename}", exist_ok=True)

        image = cv2.imread(imgp)
        qrcode = cv2.imread(qrcp)

        T_b, T_w = 50, 200  # thresholds for low and high gray levels
        polarized_image = histogram_polarization(image, T_b, T_w, basename)
        cv2.imwrite(f'./output/{basename}/Ihc.jpg', polarized_image)