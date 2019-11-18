#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

def task_1_3_apply_filters(filename1, filename2):
    color_raw = o3d.io.read_image("./data/image00834.png")
    depth_raw = o3d.io.read_image("./data/depth00834.png")

    color_gaussian = cv.GaussianBlur(np.asarray(color_raw), (3, 3), 0)
    depth_gaussian = cv.GaussianBlur(np.asarray(depth_raw), (3, 3), 0)

    color_sobel = cv.Sobel(np.asarray(color_raw), -1, 1, 1)
    depth_sobel = cv.Sobel(np.asarray(depth_raw), -1, 1, 1)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    rgbd_image_gaussian = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_gaussian), o3d.geometry.Image(depth_gaussian))

    plt.subplot(1, 2, 1)
    plt.imshow(depth_raw)
    plt.subplot(1, 2, 2)
    plt.imshow(depth_gaussian)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])

    pcd_gaussian = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_gaussian,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )
    pcd_gaussian.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    o3d.visualization.draw_geometries([pcd_gaussian])

    return

if __name__ == "__main__":
    task_1_3_apply_filters("a", "b")