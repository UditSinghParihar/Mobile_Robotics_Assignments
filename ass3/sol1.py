#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import glob
np.set_printoptions(threshold=sys.maxsize)

# IMPORTANT: Directions to use code written in report.
# Given

B = 0.53790448812
K = np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02], 
              [0.000000e+00, 7.070912e+02, 1.831104e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])
f = K[0,0] #focal length
cx= K[0,2]
cy= K[1,2]
#NOTE: Change path here
Rt_GT = np.fromfile("./a3_data/poses.txt", dtype=float, count=-1, sep=" ")
Rt_GT = np.reshape(Rt_GT, (-1,3,4))

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    # saves point clouds as ply files.
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# Step 1 and 2: Disparity calculation
def disparity(left_img, right_img):
    # If you want to use grayscale images, change function to BM_create
    window_size = 5
    min_disp = -1 #16 
    num_disp = 63-min_disp #64
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,#15
        P1 = 8*3*window_size**2,
        P2 = 16*3*window_size**2,
        disp12MaxDiff = -1,
        uniquenessRatio = 10,
        speckleWindowSize = 5,
        speckleRange = 5 
    ) 
#    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
    disparity_val = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity_val

# X, Y, Z with respect to left camera: X_L, Y_L, Z_L
def XYZfromDisparity(disparity_val, B, K, left_img):
# Step 3a: Depth calculation from disparity
    disparity_flat = disparity_val.flatten()
    Z_L = np.divide(B*K[0,0], disparity_flat)
    Z_L_T = np.reshape(Z_L, (1, Z_L.shape[0]))
    disparity_flat_s = np.reshape(disparity_flat, (1, disparity_flat.shape[0]))
    #print(disparity_flat.shape, Z_L_T.shape)
# Step 3b: XYZ calulation
    row, col = np.indices(left_img.shape[0:2])
    row_flat = row.flatten()
    col_flat = col.flatten()
    Img_indices = np.stack((row_flat.T, col_flat.T), axis = 1)
    Img_indices_homo = np.append(Img_indices, np.ones((Img_indices.shape[0],1)), axis=1)
    # Camera equation but inverted: Note: It should theoretically be Z_L_T instead of disparity_flat_s, but this works better somehow during implementation.
    XYZ_L = np.multiply(disparity_flat_s, np.linalg.inv(K) @ Img_indices_homo.T) 
    #XYZ_L = np.multiply(Z_L_T, np.linalg.inv(K) @ Img_indices_homo.T) 
    return XYZ_L

# making point cloud, visualization and saving into ply file.
def custom_pcd(XYZ_L, colors, i):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ_L.T) #numpy_points is your Nx3 cloud
    pcd.colors = o3d.utility.Vector3dVector(colors) #numpy_colors is an Nx3 matrix with the corresponding RGB colors

    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("cloud_custom" + str(i) + ".ply", pcd)
    print("custom cloud " + str(i) + " saved")

imgs_L = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in sorted(glob.glob("./a3_data/img2/*.png"))]
imgs_L_gs = [cv2.imread(file, 0) for file in sorted(glob.glob("./a3_data/img2/*.png"))] #grayscale
imgs_R = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in sorted(glob.glob("./a3_data/img3/*.png"))]
imgs_R_gs = [cv2.imread(file, 0) for file in sorted(glob.glob("./a3_data/img3/*.png"))] #grayscale

Q = np.float32([[1,0,0,0], #cx
    [0,-1,0,0], #cy giving bad result
    [0,0,f*0.03,0],
    [0,0,0,1]])

sample_size = 3
#for i in range(len(imgs_L)):
for i in range(sample_size):
    disparity_val = disparity(imgs_L[i], imgs_R[i]) # Tune disparity parameters.
    #plt.imshow(disparity_val, 'gray')
    #plt.show()

    XYZ_L = XYZfromDisparity(disparity_val, B, K, imgs_L[i])
    numpy_colors = imgs_L[i].reshape((imgs_L[i].shape[0]*imgs_L[i].shape[1], imgs_L[i].shape[2])) / 255.0
    #FUNCTION: Making point cloud, visualization and saving into ply file.
    custom_pcd(XYZ_L, numpy_colors, i)

    #OpenCV Reprojection to 3D

    points_3D = cv2.reprojectImageTo3D(disparity_val, Q)
    mask_map = disparity_val > disparity_val.min()
    output_points = points_3D[mask_map]
    output_colors = imgs_L[i][mask_map]

    write_ply('cloud_opencv' + str(i) + '.ply', output_points, output_colors)
    #write_ply('out_custom4.ply', XYZ_L.T, numpy_colors)
    print("opencv cloud " + str(i) + " saved")

pcd_opencv = [o3d.io.read_point_cloud(file) for file in sorted(glob.glob("./cloud_opencv*.ply"))]
pcd_custom = [o3d.io.read_point_cloud(file) for file in sorted(glob.glob("./cloud_custom*.ply"))]

# Transforming points in world frame
#for j in range(len(imgs_L)):
for j in range(sample_size):
    # OpenCV
    X_c = (np.asarray(pcd_opencv[j].points))
    X_c_homo = np.append(X_c, np.ones((X_c.shape[0],1)), axis=1)
    Rt_GT_homo = np.append(Rt_GT[j], np.array([[0,0,0,1]]), axis=0)
    X_w = Rt_GT_homo @ X_c_homo.T # Transforming points in camera frame to world frame.
    X_w_0 = (np.delete(X_w.T, -1, 1))
    pcd_opencv[j].points = o3d.utility.Vector3dVector(X_w_0) #pcd_opencv[0].points 
    #pcd_opencv[j].colors = pcd_opencv[j].colors
## NOTE: Uncomment lines 153 to 158 and 165 to see custom 3D projection visualization.
   # Custom
#    X_c_custom = (np.asarray(pcd_custom[j].points))
#    X_c_homo_custom = np.append(X_c_custom, np.ones((X_c_custom.shape[0],1)), axis=1)
#    X_w_custom = Rt_GT_homo @ X_c_homo_custom.T
#    X_w_0_custom = (np.delete(X_w_custom.T, -1, 1))
#    pcd_custom[j].points = o3d.utility.Vector3dVector(X_w_0_custom) #pcd_opencv[0].points 
#    pcd_custom[j].colors = o3d.utility.Vector3dVector(np.asarray(pcd_custom[j].colors))# * 255.0)

#o3d.visualization.draw_geometries([pcd_opencv[k] for k in range(len(imgs_L))])
print("You are seeing opencv implemention's cloud now.")
o3d.visualization.draw_geometries([pcd_opencv[k] for k in range(sample_size)])
print("You are seeing our custom implementation's cloud now. (Uncomment the next line to see this)")
# o3d.visualization.draw_geometries([pcd_custom[k] for k in range(sample_size)])
