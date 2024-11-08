import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.ndimage import gaussian_filter
import argparse


def edge_blending(wrap_img, H):
    inv_H = np.linalg.inv(H)
    for i in range(wrap_img.shape[0]):
        for j in range(wrap_img.shape[1]):
            coor = np.array([j, i, 1])
            img_right_coor = inv_H @ coor  # the coor of right image
            img_right_coor /= img_right_coor[2]

            # Interpolation
            y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))

            if x == 0:
                # Apply Gaussian filter for edge blending
                wrap_img[i, j] = gaussian_filter(wrap_img[i, j], sigma=2)
    return wrap_img


def wrapping(img1, img2, H):
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    wrap_img = np.zeros((max(h1, h2), w1+w2, 3), dtype="int")
    wrap_img[:h1, :w1] = img1

    # Transform left corr to right, and reproject to warped image
    inv_H = np.linalg.inv(H)
    for i in range(wrap_img.shape[0]):
        for j in range(wrap_img.shape[1]):
            coor = np.array([j, i, 1])  #
            img_right_coor = inv_H @ coor  # the coor of right image
            img_right_coor /= img_right_coor[2]

            # interpolation
            y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))
            if (x < 0 or x >= h2 or y < 0 or y >= w2):
                continue
            # wrap pixel
            # wrap_img[i, j] = img2[x, y]
            alpha = max(0, min(1, (w1 - j) / float(w1)))
            if (wrap_img[i, j][0] == 0) and (wrap_img[i, j][1] == 0) and (wrap_img[i, j][2] == 0):
                wrap_img[i, j] = img2[x, y]
            else:
                wrap_img[i, j] = (wrap_img[i, j] * alpha +
                                  img2[x, y] * (1 - alpha)).astype(int)

    wrap_img = edge_blending(wrap_img, H)
    return wrap_img


def save_match(image1, image2, keypoints1, keypoints2, matches, name):
    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    [cv2.DMatch(
                                        _queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0) for i, j in matches],
                                    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert BGR to RGB for displaying with matplotlib
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    plt.imsave(f"{name}_match.jpg", matched_image_rgb)


def homomat(points_in_img1, points_in_img2, sampling_point=4, threshold=5, S=1500):
    max_inliers = 0
    best_H = None

    for _ in range(S):
        if len(points_in_img1) > sampling_point:
            sample_indices = np.random.choice(
                len(points_in_img1), sampling_point, replace=False)
            pts1 = points_in_img1[sample_indices]
            pts2 = points_in_img2[sample_indices]
        else:
            pts1 = points_in_img1
            pts2 = points_in_img2

        H = homography(pts1, pts2)

        inliers = 0
        for (x1, y1), (x2, y2) in zip(points_in_img1, points_in_img2):
            pred = np.dot(H, np.array([x1, y1, 1]))
            pred /= pred[2]  # normalize
            error = np.linalg.norm(np.array([x2, y2]) - pred[:2])
            if error < threshold:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H
    return best_H


def homography(pts1, pts2):
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2]   # Normalize to ensure H[2,2] is 1
    return H


def get_keypoint(img1, img2, method='SIFT'):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    if method == 'SIFT':
        detector = cv2.SIFT_create()
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    elif method == 'MSER':
        detector = cv2.MSER_create()
        keypoints1 = detector.detect(img1)
        keypoints2 = detector.detect(img2)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.compute(img1, keypoints1)
        keypoints2, descriptors2 = sift.compute(img2, keypoints2)
    elif method == 'HARRIS':
        keypoints1 = harris_corner_detector(img1)
        keypoints2 = harris_corner_detector(img2)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.compute(img1, keypoints1)
        keypoints2, descriptors2 = sift.compute(img2, keypoints2)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'SIFT', 'MSER', or 'HARRIS'.")

    img1_keypoints = cv2.drawKeypoints(
        img1, keypoints1, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(
        img2, keypoints2, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    combined_img = cv2.hconcat([img1_keypoints, img2_keypoints])
    cv2.imwrite('img_keypoint.jpg', combined_img)
    print("img_keypoint.jpg saved")

    return [keypoints1, descriptors1], [keypoints2, descriptors2]


def harris_corner_detector(img):
    dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    keypoints = []
    threshold = 0.01 * dst.max()
    for y in range(dst.shape[0]):
        for x in range(dst.shape[1]):
            if dst[y, x] > threshold:
                keypoints.append(cv2.KeyPoint(x, y, 1))
    return keypoints


def match_keypoints(descriptors1, descriptors2, threshold=0.5):
    best_score_pairs = []
    best_index_pairs = []
    for i in range(len(descriptors1)):
        best_scores = [float('inf'), float('inf')]
        best_index = [0, 0]
        for j in range(len(descriptors2)):
            score = np.linalg.norm(descriptors1[i] - descriptors2[j])
            if score < best_scores[0]:
                best_scores[1] = best_scores[0]
                best_index[1] = best_index[0]
                best_scores[0] = score
                best_index[0] = j
            elif score < best_scores[1]:
                best_scores[1] = score
                best_index[1] = j
        best_score_pairs.append(best_scores)
        best_index_pairs.append(best_index)

    final_index_pairs = []
    for i in range(len(best_index_pairs)):
        score = best_score_pairs[i][0] / best_score_pairs[i][1]
        if score < threshold:
            final_index_pairs.append((i, best_index_pairs[i][0]))
    print("Length of good match:", len(final_index_pairs))

    return final_index_pairs


def stitch_images(image1, image2, name, method='SIFT', sampling_point=4, threshold=0.5):
    key1, key2 = get_keypoint(image1, image2, method=method)
    key_index = match_keypoints(key1[1], key2[1], threshold)

    points_in_img1 = np.float32([key1[0][i].pt for i, _ in key_index])
    points_in_img2 = np.float32([key2[0][j].pt for _, j in key_index])

    H = homomat(points_in_img2, points_in_img1,
                sampling_point, S=1500, threshold=5)
    warped_image = wrapping(image1, image2, H)
    cv2.imwrite(f"{name}_{method}.jpg", warped_image)
    save_match(image1, image2, key1[0], key2[0], key_index, f"{name}_{method}")


def main():
    parser = argparse.ArgumentParser(
        description='Image Stitching with different feature detection methods.')
    parser.add_argument(
        '--image1', type=str, default='data/hill1.JPG', help='Path to the first image')
    parser.add_argument(
        '--image2', type=str,  default='data/hill2.JPG', help='Path to the second image')
    parser.add_argument('--output_name', type=str, default='hill',
                        help='Name for the output stitched image')
    parser.add_argument('--method', type=str, default='SIFT', choices=['SIFT',  'HARRIS', 'MSER'],
                        help='Feature detection method to use for stitching (default: SIFT)')
    parser.add_argument('--sampling_point', type=int, default=4,
                        help='Number of sampling points for homography estimation (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for feature matching (default: 0.5)')

    args = parser.parse_args()

    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)

    stitch_images(image1, image2, args.output_name, method=args.method,
                  sampling_point=args.sampling_point, threshold=args.threshold)


if __name__ == "__main__":
    main()
