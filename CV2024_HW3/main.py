import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def sift_feature_detection(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2

def feature_mapping(des1, des2, ratio=0.75):
    matches = []
    for i, des1i in enumerate(des1):
        distances = np.linalg.norm(des2 - des1i, axis=1)
        min_idx = np.argpartition(distances, 2)[:2]
        min_dis, second_min_dis = distances[min_idx[0]], distances[min_idx[1]]

        if min_dis < ratio*second_min_dis:
            matches.append((i, min_idx[0]))
    return matches

def homomat(points_in_img1, points_in_img2, threshold=5):
    max_inliers = 0
    best_H = None

    for _ in range(1000):
        sample_indices = np.random.choice(len(points_in_img1), 4, replace=False)
        pts1 = points_in_img1[sample_indices]
        pts2 = points_in_img2[sample_indices]
        
        H = homography(pts1, pts2)

        inliers = 0
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            pred = np.dot(H, np.array([x1, y1, 1]))
            pred /= pred[2] # normalize
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

# Step 4: Image Wrapping and Blending
def wrap(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find corners of img1 and map them to the new position using H
    corners_img1 = np.array([
        [0, 0, 1],        # Top-left corner
        [w1, 0, 1],       # Top-right corner
        [0, h1, 1],       # Bottom-left corner
        [w1, h1, 1]       # Bottom-right corner
    ])

    # Transform corners of img1 to the coordinates in img2
    transformed_corners = (H @ corners_img1.T).T
    transformed_corners /= transformed_corners[:, 2:3]  # Normalize by the third coordinate

    # Calculate bounds for the new image
    min_x = int(min(transformed_corners[:, 0].min(), 0))
    max_x = int(max(transformed_corners[:, 0].max(), w2))
    min_y = int(min(transformed_corners[:, 1].min(), 0))
    max_y = int(max(transformed_corners[:, 1].max(), h2))

    # Create the output image with the determined size
    out_width = max_x - min_x
    out_height = max_y - min_y
    result = np.zeros((out_height, out_width, 3), dtype=img1.dtype)

    # Offset to shift img2 to the new position in the output image
    offset_x = -min_x
    offset_y = -min_y

    # Place img2 in the output image
    result[offset_y:offset_y+h2, offset_x:offset_x+w2] = img2

    # Inverse the homography to map each pixel in the output to img1's coordinates
    H_inv = np.linalg.inv(H)

    # Iterate over every pixel in the output image and map it back to img1
    for y in range(out_height):
        for x in range(out_width):
            # Map the pixel in the output image back to img1's coordinate system
            pt = np.array([x - offset_x, y - offset_y, 1])
            transformed_pt = H_inv @ pt
            transformed_pt /= transformed_pt[2]  # Normalize

            # Get the x, y coordinates in img1
            src_x, src_y = int(transformed_pt[0]), int(transformed_pt[1])

            # Check if the coordinates are within img1's bounds
            if 0 <= src_x < w1 and 0 <= src_y < h1:
                # Simple blending: here we just overwrite pixels; you could implement alpha blending here
                result[y, x] = img1[src_y, src_x]

    return result
    # h1, w1 = img1.shape[:2]
    # h2, w2 = img2.shape[:2]
    
    # # Get corners of img2 and transform them with H to find the output size
    # corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    # warped_corners = (H @ np.concatenate([corners_img2.squeeze(), np.ones((4, 1))], axis=1).T).T
    # warped_corners = (warped_corners[:, :2] / warped_corners[:, 2:]).astype(int)

    # # Determine dimensions of the output image
    # min_x = min(warped_corners[:, 0].min(), 0)
    # min_y = min(warped_corners[:, 1].min(), 0)
    # max_x = max(warped_corners[:, 0].max(), w1)
    # max_y = max(warped_corners[:, 1].max(), h1)

    # # Calculate translation matrix
    # translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    # output_shape = (max_y - min_y, max_x - min_x)
    
    # # Create the output image
    # result = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
    
    # # Apply inverse transformation to img2 pixels and blend with img1
    # for y in range(output_shape[0]):
    #     for x in range(output_shape[1]):
    #         # Apply the inverse transform
    #         src_pos = np.linalg.inv(translation @ H) @ np.array([x, y, 1])
    #         src_x, src_y = src_pos[:2] / src_pos[2]
    #         src_x, src_y = int(round(src_x)), int(round(src_y))
            
    #         if 0 <= src_x < w2 and 0 <= src_y < h2:
    #             result[y, x] = img2[src_y, src_x]
    
    # # Place img1 onto the canvas
    # result[-min_y:h1-min_y, -min_x:w1-min_x] = img1
    
    # # Simple alpha blending for overlapping regions
    # # overlap = (result > 0) & (img1 > 0)
    # # result[overlap] = result[overlap] * 0.5 + img1[overlap] * 0.5

    # return result

def stitch_images(img1, img2, gray1, gray2, i):
    kp1, des1, kp2, des2 = sift_feature_detection(gray1, gray2)
    plot_keypoints(gray1, gray2, kp1, kp2, i)
    matches = feature_mapping(des1, des2)
    plot_matches(img1, img2, kp1, kp2, matches, i)

    # Get point coordinates based on matches
    points_in_img1 = np.float32([kp1[i].pt for i, _ in matches])
    points_in_img2 = np.float32([kp2[j].pt for _, j in matches])

    # Calculate homography
    H = homomat(points_in_img1, points_in_img2)

    # Wrap images
    result = wrap(img1, img2, H)
    return result

def plot_keypoints(img1, img2, kp1, kp2, i):
    img1 = cv2.drawKeypoints(img1, kp1, img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    img = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(os.path.join('output/keypoints', f'keypoints_{i//2}.jpg'), img)

def plot_matches(img1, img2, kp1, kp2, matches, x):
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    img1_with_border = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    img1_with_border[:img1.shape[0], :img1.shape[1]] = img1
    img1_with_border[:img2.shape[0], img1.shape[1]:] = img2
    
    img1_with_border = cv2.cvtColor(img1_with_border, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    plt.imshow(img1_with_border)
    
    for i, j in matches:
        pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
        pt2 = (int(kp2[j].pt[0] + img1.shape[1]), int(kp2[j].pt[1]))
        random_color = (random.random(), random.random(), random.random())
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], random_color, linewidth=0.5)

    plt.axis('off')
    plt.savefig(os.path.join('output/matches', f'matches_{x//2}.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == '__main__':
    img_path = glob.glob('data/*')
    for i in range(0, len(img_path)-1, 2):
        print(f'processing image {i//2 + 1}...')
        img1 = cv2.imread(img_path[i])
        img2 = cv2.imread(img_path[i+1])
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        result = stitch_images(img1, img2, gray1, gray2, i)
        cv2.imwrite(os.path.join('output', f'stitched_image_{i//2}.jpg'), result)