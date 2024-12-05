import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from read_file import read_intrinsics

import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import open3d.visualization.rendering as rendering
from scipy.spatial import Delaunay

def extract_keypoint(img):
    sift = cv2.SIFT_create()
    keypoint, descriptors = sift.detectAndCompute(img, None)
    return keypoint, descriptors

def match_keypoint(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def cal_fundamental_matrix(key1, key2, matches, mode):
    pts1 = np.float32([key1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([key2[m.trainIdx].pt for m in matches])
    if mode == 'ransac':
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99) # mask is selected by RANSAC
    else:
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
    return F, mask, pts1, pts2

def draw_epilines(img, lines):
    r, c = img.shape[:2]
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        img_color = cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
    return img_color

def E_to_pose(E):
    # eigenvalue correction : eigenvalue of E should have two and have same value
    U, S, Vt = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    E = U @ np.diag([m, m, 0]) @ Vt

    # calculate rotation matrix and translation(t) using svd
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # rotation matrix
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return R1, R2, t

def find_best_rt_and_triangulate(R1, R2, t, pts1, pts2, K1, K2):
    possible_poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    best_pose = None
    max_points_in_front = 0
    best_points_3d = None

    for R, t in possible_poses:
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # K1 [I|0]
        P2 = K2 @ np.hstack((R, t.reshape(3, 1)))           # K2 [R|t]
        pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # 2D point to homo
        pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T  # 2D point to homo

        tri_points_3D_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
        tri_points_3D = tri_points_3D_h[:3] / tri_points_3D_h[3]

        points_in_front = np.sum(tri_points_3D[2] > 0)

        if points_in_front > max_points_in_front:
            max_points_in_front = points_in_front
            best_pose = (R, t)
            best_points_3d = tri_points_3D.T

    return best_pose, best_points_3d

def create_point_cloud(points_3d, img, keypoints):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the point cloud with the 3D points
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Extract the color information from the image based on the keypoints
    colors = []
    for pt in keypoints:
        x, y = int(pt.pt[0]), int(pt.pt[1])
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            print(img.shape)
            b, g, r = img[y, x]  # Extract BGR values
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Convert to RGB and normalize to [0, 1]
        else:
            colors.append([0.5, 0.5, 0.5])  # Default color if keypoint is out of bounds
    
    # Set the colors in the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def create_textured_mesh(points_3d, img, triangles):
    # Create a TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()

    # Set the vertices of the mesh with the 3D points
    mesh.vertices = o3d.utility.Vector3dVector(points_3d)

    # Set the triangles (faces) of the mesh
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Extract color information from the image based on the keypoints
    vertex_colors = []
    for pt in points_3d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            b, g, r = img[y, x]  # Extract BGR values
            vertex_colors.append([r / 255.0, g / 255.0, b / 255.0])  # Convert to RGB and normalize to [0, 1]
        else:
            vertex_colors.append([0.5, 0.5, 0.5])  # Default color if vertex is out of bounds
    
    # Set the vertex colors in the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return mesh

def save_mesh_with_texture(mesh, image_path, output_obj_path):
    # Load the texture image
    texture_image = cv2.imread(image_path)
    if texture_image is None:
        raise FileNotFoundError(f"Texture image not found at: {image_path}")
    
    # Normalize vertices to compute UVs
    vertices = np.asarray(mesh.vertices)
    uvs = np.zeros((vertices.shape[0], 2))  # Allocate space for UVs
    uvs[:, 0] = (vertices[:, 0] - np.min(vertices[:, 0])) / (np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    uvs[:, 1] = (vertices[:, 1] - np.min(vertices[:, 1])) / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    # Open3D expects UVs normalized in [0, 1]
    uvs = np.clip(uvs, 0, 1)

    # Assign UVs to mesh
    triangles = np.asarray(mesh.triangles)
    flattened_uvs = uvs[triangles].reshape(-1, 2)  # Flatten UVs
    mesh.triangle_uvs = o3d.utility.Vector2dVector(flattened_uvs)

    # Assign the texture image
    mesh.textures = [o3d.geometry.Image(cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB))]

    # Save mesh with texture
    o3d.io.write_triangle_mesh(output_obj_path, mesh, write_triangle_uvs=True)
    print(f"Textured mesh saved to: {output_obj_path}")



def main(img1_path, img2_path, txt_path, savedir, mode, show_3d=False):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1_rgb = cv2.imread(img1_path)
    img2_rgb = cv2.imread(img2_path)
    K1, K2 = read_intrinsics(txt_path)

    # key points matching
    key1, des1 = extract_keypoint(img1)
    key2, des2 = extract_keypoint(img2)
    matches = match_keypoint(des1, des2)
    interest_point_image = cv2.drawKeypoints(img2, key2, 0, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    matched_image = cv2.drawMatches(img1, key1, img2, key2, matches[:150], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f'{savedir}/{mode}/interest_point_image.jpg', interest_point_image)
    cv2.imwrite(f'{savedir}/{mode}/match_image.jpg', matched_image)

    F, mask, pts1, pts2 = cal_fundamental_matrix(key1, key2, matches, mode)
    E = K2.T @ F @ K1
    pts1_inliers = pts1[mask.ravel() == 1].reshape(-1, 2)
    pts2_inliers = pts2[mask.ravel() == 1].reshape(-1, 2)
    print("\" Fundamental Matrix \"")
    print(F)

    # calculate epipolar line 
    # (ref: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html)
    lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F) # epiline in left image
    lines1 = lines1.reshape(-1, 3)
    img1_epilines = draw_epilines(img1, lines1)
    cv2.imwrite(f'{savedir}/{mode}/epilines_image1.jpg', img1_epilines)

    lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F) # epiline in right image
    lines2 = lines2.reshape(-1, 3)
    img2_epilines = draw_epilines(img2, lines2)
    cv2.imwrite(f'{savedir}/{mode}/epilines_image2.jpg', img2_epilines)

    # Calculate R and t from E
    R1, R2, t = E_to_pose(E)
    print("\n\" Rotation Matrix \"")
    print(R1)
    print(R2)
    print(t)

    # Triangulate
    best_pose, points_3d = find_best_rt_and_triangulate(R1, R2, t, pts1_inliers, pts2_inliers, K1, K2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.title('3D Points Triangulated')
    # plt.show()

    # Create point cloud
    pcd = create_point_cloud(points_3d, img1_rgb, key1)
    
    # Create a simple triangulation using Delaunay triangulation
    tri = Delaunay(points_3d[:, :2])
    triangles = tri.simplices
    
    # Create textured mesh
    mesh = create_textured_mesh(points_3d, img1_rgb, triangles)

    # Visualize point cloud and textured mesh
    if show_3d:
        o3d.visualization.draw_geometries([pcd], window_name='3D Point Cloud from SFM', width=800, height=600)
        o3d.visualization.draw_geometries([mesh], window_name='Textured 3D Mesh from SFM', width=800, height=600)
    
    # Save point cloud and mesh for future use
    o3d.io.write_point_cloud(f'{savedir}/point_cloud/sfm_3d_model.ply', pcd)
    o3d.io.write_triangle_mesh(f'{savedir}/mesh/sfm_textured_mesh.obj', mesh)
    
    print("3D model saved as 'sfm_3d_model.ply'")
    print("Textured mesh saved as 'sfm_textured_mesh.obj'")

    print("Finished texture mapping, point cloud, and mesh creation.")

    # create .mtl file
    save_mesh_with_texture(mesh, img1_path, f'{savedir}/mesh/sfm_textured_mesh.obj')


if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    show_3d = True

    datapath = glob.glob('data/*') + glob.glob('my_data/*')
    for i in range(0, len(datapath), 3):
        basename = os.path.basename(datapath[i+2]).split('_')[0]
        os.makedirs(f"output/{basename}/ransac", exist_ok=True)
        os.makedirs(f"output/{basename}/8-point", exist_ok=True)
        os.makedirs(f"output/{basename}/point_cloud", exist_ok=True)
        os.makedirs(f"output/{basename}/mesh", exist_ok=True)

        savedir = f"./output/{basename}"
        main(datapath[i], datapath[i+1], datapath[i+2], savedir, '8-point', show_3d)
        main(datapath[i], datapath[i+1], datapath[i+2], savedir, 'ransac', show_3d)
