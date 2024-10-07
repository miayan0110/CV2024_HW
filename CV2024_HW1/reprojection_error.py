import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
corner_x = 10
corner_y = 7
objp = np.zeros((corner_x * corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('my_data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    # Find the chessboard corners
    print('find the chessboard corners of', fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
        plt.imshow(img)

# Camera calibration
print('Camera calibration...')
img_size = (gray.shape[1], gray.shape[0])  # Image size (width, height)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)  # OpenCV calibration


def calculate_camera_parameters(objpoints, imgpoints):
    """
    計算相機內部參數和外部參數。
    參數：
    objpoints: 3D 點的列表（世界坐標系）
    imgpoints: 2D 點的列表（圖像坐標系）
    返回：
    K: 相機內部參數矩陣
    extrinsics: 相機外部參數（旋轉矩陣和位移向量）
    """

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
        h_i = h_i / h_i[-1]  # consider scale coef
        H.append(h_i.reshape((3, 3)))
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

    return extrinsics


extrinsics_custom = calculate_camera_parameters(objpoints, imgpoints)


def calculate_camera_parameters_open_cv(rvecs, tvecs):
    extrinsics = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        extrinsics.append(np.column_stack((R, tvec)))
    return np.array(extrinsics)


extrinsics_opencv = calculate_camera_parameters_open_cv(rvecs, tvecs)


def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, extrinsics):
    print('Calculating reprojection error...')
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        # Extract rotation and translation vectors from extrinsics
        rvec, tvec = extrinsics[i][:, :3], extrinsics[i][:, 3]

        # Project points using the camera calibration parameters and extrinsics
        projected_image_points, _ = cv2.projectPoints(
            objpoints[i], rvec, tvec, mtx, dist)

        # Compute the error between the detected points and projected points
        error = cv2.norm(imgpoints[i], projected_image_points,
                         cv2.NORM_L2) / len(projected_image_points)
        print("Image: ", i, "Error: ", error)
        total_error += error
        total_points += len(objpoints[i])

    # Compute average reprojection error
    mean_error = total_error / len(objpoints)
    return mean_error


# 計算兩種方法的重投影誤差
mean_error_custom = calculate_reprojection_error(
    objpoints, imgpoints, mtx, dist, extrinsics_custom)
print(f"Mean Reprojection Error (Custom): {mean_error_custom}")

mean_error_opencv = calculate_reprojection_error(
    objpoints, imgpoints, mtx, dist, extrinsics_opencv)
print(f"Mean Reprojection Error (OpenCV): {mean_error_opencv}")
