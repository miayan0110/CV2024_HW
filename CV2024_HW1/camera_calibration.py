import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)
    
    # if idx == 3:
    #     break
# plt.show()

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None) # opencv
"""
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs) # rotation vector
Tr = np.array(tvecs) # translation vector
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""

"""
Write your code here
"""
################################################################################################################################
"""Calculate H"""
def get_Ai(obj_ps, img_ps):
    A_i = []
    for obj_p, img_p in zip(obj_ps, img_ps):
        X = obj_p[0]
        Y = obj_p[1]
        u = img_p[0][0]
        v = img_p[0][1]
        A_row1 = [X, Y, 1, 0, 0, 0, -X * u, -Y * u, -u]
        A_row2 = [0, 0, 0, X, Y, 1, -X * v, -Y * v, -v]
        A_i.append(A_row1)
        A_i.append(A_row2)
    return np.array(A_i)

# calculate H
H = []
for i in range(0, len(objpoints)):
    A_i = get_Ai(objpoints[i], imgpoints[i])
    u, s, vt = np.linalg.svd(A_i)
    h_i = vt[-1, :]
    h_i = h_i / h_i[-1] # consider scale coef
    H.append(h_i.reshape((3,3)))
H = np.array(H)
# =================================================================
"""Calculate Intrinsic"""
# calculate V
def v_pq(H_i, p, q):
    v = np.array([
        H_i[0, p] * H_i[0, q],
        H_i[0, p] * H_i[1, q] + H_i[1, p] * H_i[0, q],
        H_i[1, p] * H_i[1, q],
        H_i[2, p] * H_i[0, q] + H_i[0, p] * H_i[2, q],
        H_i[2, p] * H_i[1, q] + H_i[1, p] * H_i[2, q],
        H_i[2, p] * H_i[2, q]
    ])
    return v

def get_V(H):
    V = []
    for H_i in H:
        v12 = v_pq(H_i, 0, 1)
        v11 = v_pq(H_i, 0, 0)
        v22 = v_pq(H_i, 1, 1)

        V.append(v12)
        V.append((v11 - v22))
    return np.array(V)
V = get_V(H)

# calculate b (b11, b12, b13, b22, b23, b33)
u, s, vt = np.linalg.svd(V)
b = vt[-1, :]
b11, b12, b22, b13, b23, b33 = b[0], b[1], b[2], b[3], b[4], b[5]

# calculate intrinsics
o_y = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
lamb = b33 - (b13**2 + o_y * (b12 * b13 - b11 * b23)) / b11
alpha = np.sqrt(lamb / b11)
beta = np.sqrt(lamb * b11 / (b11 * b22 - b12**2))
gamma = -b12 * alpha**2 * beta / lamb
o_x = gamma * o_y / beta - b13 * alpha**2 / lamb

K = np.array([[alpha,   0,   o_x],
              [0,     beta,  o_y],
              [0,     0,     1]])
# =================================================================
"""Calculate Extrinsics"""
# calculate Extrinsic (R t)
extrinsics = []
for H_i in H:
    h1 = H_i[:, 0]
    h2 = H_i[:, 1]
    h3 = H_i[:, 2]
    K_inv = np.linalg.inv(K)

    lambda_ = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = lambda_ * np.dot(K_inv, h1)
    r2 = lambda_ * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))
    t = lambda_ * np.dot(K_inv, h3)

    extrinsics.append(np.column_stack((R, t)))
extrinsics = np.array(extrinsics)
################################################################################################################################

print('Show the camera extrinsics')

# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

"""
#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)


