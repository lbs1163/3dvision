#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

f_x = 525.0
f_y = 525.0
c_x = 319.5
c_y = 239.5

def task_1_3_apply_filters(file_color1, file_depth1, file_color2, file_depth2):
    def read(color, depth):
        color_raw = o3d.io.read_image(file_color1)
        depth_raw = o3d.io.read_image(file_depth1)

        grayscale_raw = cv.cvtColor(np.asarray(color_raw), cv.COLOR_BGR2GRAY)

        grayscale_gaussian = cv.GaussianBlur(grayscale_raw, (3, 3), 0)

        grayscale_sobel_dx = cv.Sobel(grayscale_gaussian, -1, 1, 0)
        grayscale_sobel_dy = cv.Sobel(grayscale_gaussian, -1, 0, 1)

        return {
            "color_raw": color_raw,
            "grayscale": {
                "raw": grayscale_raw,
                "gaussian": grayscale_gaussian,
                "sobel": {
                    "dx": grayscale_sobel_dx,
                    "dy": grayscale_sobel_dy
                }
            },
            "depth": np.asarray(depth_raw),
            "depth_raw": depth_raw,
        }

    return {"i": read(file_color1, file_depth1), "j": read(file_color2, file_depth2)};

def task_1_4_calculate_jacobian_matrix(image, T):
    I_j = image["grayscale"]["gaussian"][image["depth"] != 0]
    size = I_j.size

    [v, u] = np.indices(image["grayscale"]["gaussian"].shape)
    u = np.reshape(u[image["depth"] != 0], size)
    v = np.reshape(v[image["depth"] != 0], size)
    d = np.reshape(image["depth"][image["depth"] != 0], size)

    h_x = (u - c_x) * d / f_x
    h_y = (v - c_y) * d / f_y
    h_z = d
    h = np.stack((h_x, h_y, h_z, np.ones(size)), axis=0)

    s = T @ h
    s_x = s[0] / s[3]
    s_y = s[1] / s[3]
    s_z = s[2] / s[3]

    g_u = s_x * f_x / s_z + c_x
    g_v = s_y * f_y / s_z + c_y

    dx = np.reshape(image["grayscale"]["sobel"]["dx"][image["depth"] != 0], (size, 1))
    dy = np.reshape(image["grayscale"]["sobel"]["dy"][image["depth"] != 0], (size, 1))

    J_r = np.zeros((size, 6))

    for i in range(0, size):
        # 1 x 2
        I_g = np.transpose(np.array([dx[i], dy[i]]))

        # 2 x 3
        g_s = np.array([
            [f_x/s_z[i],            0, - (f_x * s_x[i]) / (s_z[i] ** 2)],
            [         0, f_y / s_z[i], - (f_y * s_y[i]) / (s_z[i] ** 2)],
        ])
        
        # 3 x 6
        s_xi = np.array([
            [      0,  h_z[i], -h_y[i], 1, 0, 0],
            [-h_z[i],       0,  h_z[i], 0, 1, 0],
            [ h_y[i], -h_x[i],       0, 0, 0, 1],
        ])

        J_r[i] = I_g @ g_s @ s_xi
    
    return (J_r, I_j, g_u, g_v)

def task_1_5_find_transformation_matrix(images):
    xi = np.array([0, 0, 0, 0, 0, 0])

    source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(images["j"]["color_raw"], images["j"]["depth_raw"])
    target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(images["j"]["color_raw"], images["j"]["depth_raw"])

    source = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )
    target = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )

    source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)

    T = np.identity(4)

    for i in range(0, 10):
        J_r, I_j, u, v = task_1_4_calculate_jacobian_matrix(images["j"], T)

        r = np.zeros(I_j.size)

        for j in range(0, I_j.size):
            x = round(u[i])
            y = round(v[i])

            [h, w] = images["i"]["grayscale"]["gaussian"].shape

            if x < 0 or x >= w or y > 0 or y >= h:
                I_i = 0
            else:
                I_i = images["i"][y][x]
            
            r[i] = I_i - I_j[j]

        xi = xi - np.linalg.inv(np.transpose(J_r) @ J_r) @ np.transpose(J_r) @ r

        T = np.array([
            [     1, -xi[2],  xi[1], xi[3]],
            [ xi[2],      1, -xi[0], xi[4]],
            [-xi[1],  xi[0],      1, xi[5]],
            [     0,      0,      0,     1],
        ])

        source.transform(T)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

    return T

if __name__ == "__main__":
    show_results = False

    images = task_1_3_apply_filters(
        "./data/image00834.png", "./data/depth00834.png",
        "./data/image00894.png", "./data/depth00894.png"
    )

    T = task_1_5_find_transformation_matrix(images)

    if show_results:
        plt.subplot(2, 3, 1)
        plt.imshow(images["i"]["grayscale"]["raw"])
        plt.subplot(2, 3, 2)
        plt.imshow(images["i"]["grayscale"]["gaussian"])
        plt.subplot(2, 3, 3)
        plt.imshow(images["i"]["depth"])
        plt.subplot(2, 3, 5)
        plt.imshow(images["i"]["grayscale"]["sobel"]["dx"])
        plt.subplot(2, 3, 6)
        plt.imshow(images["i"]["grayscale"]["sobel"]["dy"])
        plt.show()

        plt.subplot(2, 3, 1)
        plt.imshow(images["j"]["grayscale"]["raw"])
        plt.subplot(2, 3, 2)
        plt.imshow(images["j"]["grayscale"]["gaussian"])
        plt.subplot(2, 3, 3)
        plt.imshow(images["j"]["depth"])
        plt.subplot(2, 3, 5)
        plt.imshow(images["j"]["grayscale"]["sobel"]["dx"])
        plt.subplot(2, 3, 6)
        plt.imshow(images["j"]["grayscale"]["sobel"]["dy"])
        plt.show()