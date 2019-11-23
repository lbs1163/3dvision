#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

def task_1_3_apply_filters(color_raw_1, depth_raw_1, color_raw_2, depth_raw_2):
    def apply_filter(color_raw, depth_raw):

        grayscale_raw = cv.cvtColor(color_raw, cv.COLOR_BGR2GRAY) / 255

        grayscale_gaussian = cv.GaussianBlur(grayscale_raw, (3, 3), 0)

        grayscale_sobel_dx = cv.Sobel(grayscale_gaussian, -1, 1, 0)
        grayscale_sobel_dy = cv.Sobel(grayscale_gaussian, -1, 0, 1)

        return {
            "color": color_raw,
            "grayscale": {
                "raw": grayscale_raw,
                "gaussian": grayscale_gaussian,
                "sobel": {
                    "dx": grayscale_sobel_dx,
                    "dy": grayscale_sobel_dy
                }
            },
            "depth": depth_raw,
        }

    return {"i": apply_filter(color_raw_1, depth_raw_1), "j": apply_filter(color_raw_2, depth_raw_2)}

def task_1_4_calculate_jacobian_matrix(image, xi, f_x, f_y, c_x, c_y):
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

    T = np.array([
        [     1, -xi[2],  xi[1], xi[3]],
        [ xi[2],      1, -xi[0], xi[4]],
        [-xi[1],  xi[0],      1, xi[5]],
        [     0,      0,      0,     1],
    ])

    s = T @ h
    s_x = s[0] / s[3]
    s_y = s[1] / s[3]
    s_z = s[2] / s[3]

    g_u = s_x * f_x / s_z + c_x
    g_v = s_y * f_y / s_z + c_y

    dx = np.reshape(image["grayscale"]["sobel"]["dx"][image["depth"] != 0], size)
    dy = np.reshape(image["grayscale"]["sobel"]["dy"][image["depth"] != 0], size)

    J_r = np.zeros((size, 6))

    I_g_0_0 = dx
    I_g_0_1 = dy

    g_s_0_0 = f_x / s_z
    g_s_0_2 = - (f_x * s_x) / (s_z ** 2)
    g_s_1_1 = f_y / s_z
    g_s_1_2 = - (f_y * s_y) / (s_z ** 2)

    s_xi_0_1 = h_z
    s_xi_0_2 = -h_y
    s_xi_1_0 = -h_z
    s_xi_1_2 = h_z
    s_xi_2_0 = h_y
    s_xi_2_1 = -h_x

    I_g_s_0_0 = I_g_0_0 * g_s_0_0
    I_g_s_0_1 = I_g_0_1 * g_s_1_1
    I_g_s_0_2 = I_g_0_0 * g_s_0_2 + I_g_0_1 * g_s_1_2

    I_g_s_xi_0_0 = I_g_s_0_1 * s_xi_1_0 + I_g_s_0_2 * s_xi_2_0
    I_g_s_xi_0_1 = I_g_s_0_0 * s_xi_0_1 + I_g_s_0_2 * s_xi_2_1
    I_g_s_xi_0_2 = I_g_s_0_0 * s_xi_0_2 + I_g_s_0_1 * s_xi_1_2
    I_g_s_xi_0_3 = I_g_s_0_0
    I_g_s_xi_0_4 = I_g_s_0_1
    I_g_s_xi_0_5 = I_g_s_0_2

    J_r = np.stack((I_g_s_xi_0_0, I_g_s_xi_0_1, I_g_s_xi_0_2, I_g_s_xi_0_3, I_g_s_xi_0_4, I_g_s_xi_0_5), axis=1)
    
    return (J_r, I_j, g_u, g_v)

def task_1_5_find_transformation_matrix(images, xi, f_x, f_y, c_x, c_y):
    source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(images["j"]["color"]), o3d.geometry.Image(np.uint16(images["j"]["depth"] * 1000)))
    target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(images["i"]["color"]), o3d.geometry.Image(np.uint16(images["i"]["depth"] * 1000)))

    source = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )
    target = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )

    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #vis.add_geometry(source)
    #vis.add_geometry(target)

    for i in range(0, 1000):
        J_r, I_j, u, v = task_1_4_calculate_jacobian_matrix(images["j"], xi, f_x, f_y, c_x, c_y)

        r = np.zeros(I_j.size)

        x = np.round(u)
        y = np.round(v)

        [h, w] = images["i"]["grayscale"]["gaussian"].shape

        for j in range(0, I_j.size):
            if x[j] < 0 or x[j] >= w or y[j] < 0 or y[j] >= h:
                I_i = 0
            else:
                I_i = images["i"]["grayscale"]["gaussian"][int(y[j])][int(x[j])]
            
            r[j] = I_i - I_j[j]
        
        xi = xi - np.linalg.inv(np.transpose(J_r) @ J_r) @ np.transpose(J_r) @ r

        print(xi)
        print(np.sum(r ** 2))

        #source.transform(inverse)
        #source.transform(T)
        #vis.update_geometry()
        #vis.poll_events()
        #vis.update_renderer()

    return xi

def task_1_6_build_image_pyramid(file_color1, file_depth1, file_color2, file_depth2):
    image_pyramids = [0, 0, 0]

    color_raw_1 = np.asarray(o3d.io.read_image(file_color1))
    depth_raw_1 = np.asarray(o3d.io.read_image(file_depth1)) / 1000
    color_raw_2 = np.asarray(o3d.io.read_image(file_color2))
    depth_raw_2 = np.asarray(o3d.io.read_image(file_depth2)) / 1000

    image_pyramids[2] = task_1_3_apply_filters(color_raw_1, depth_raw_1, color_raw_2, depth_raw_2)

    color_raw_1 = cv.resize(color_raw_1, dsize=(320, 240), interpolation=cv.INTER_LINEAR)
    depth_raw_1 = cv.resize(depth_raw_1, dsize=(320, 240), interpolation=cv.INTER_LINEAR) / 2
    color_raw_2 = cv.resize(color_raw_2, dsize=(320, 240), interpolation=cv.INTER_LINEAR)
    depth_raw_2 = cv.resize(depth_raw_2, dsize=(320, 240), interpolation=cv.INTER_LINEAR) / 2

    image_pyramids[1] = task_1_3_apply_filters(color_raw_1, depth_raw_1, color_raw_2, depth_raw_2)

    color_raw_1 = cv.resize(color_raw_1, dsize=(160, 120), interpolation=cv.INTER_LINEAR)
    depth_raw_1 = cv.resize(depth_raw_1, dsize=(160, 120), interpolation=cv.INTER_LINEAR) / 2
    color_raw_2 = cv.resize(color_raw_2, dsize=(160, 120), interpolation=cv.INTER_LINEAR)
    depth_raw_2 = cv.resize(depth_raw_2, dsize=(160, 120), interpolation=cv.INTER_LINEAR) / 2

    image_pyramids[0] = task_1_3_apply_filters(color_raw_1, depth_raw_1, color_raw_2, depth_raw_2)

    return image_pyramids

if __name__ == "__main__":
    show_results = False

    f_x = 525.0
    f_y = 525.0
    c_x = 319.5
    c_y = 239.5

    image_pyramids = task_1_6_build_image_pyramid(
        "./data/image00834.png", "./data/depth00834.png",
        "./data/image00894.png", "./data/depth00894.png"
    )

    xi = np.array([0, 0, 0, 0, 0, 0])

    for i in range(0, 3):
        xi = task_1_5_find_transformation_matrix(image_pyramids[i], xi, f_x / (2 ** (2 - i)), f_y / (2 ** (2 - i)), c_x / (2 ** (2 - i)), c_y / (2 ** (2 - i)))

        T = np.array([
            [     1, -xi[2],  xi[1], xi[3]],
            [ xi[2],      1, -xi[0], xi[4]],
            [-xi[1],  xi[0],      1, xi[5]],
            [     0,      0,      0,     1],
        ])

        source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image("./data/image00834.png"), o3d.io.read_image("./data/depth00834.png"))
        target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image("./data/image00894.png"), o3d.io.read_image("./data/depth00894.png"))\
        
        source = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )
        target = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )

        T = np.array([
            [     1, -xi[2],  xi[1], xi[3]],
            [ xi[2],      1, -xi[0], xi[4]],
            [-xi[1],  xi[0],      1, xi[5]],
            [     0,      0,      0,     1],
        ])

        source.transform(T)

        T = np.array([
            [     1, -xi[2],  xi[1], xi[3]],
            [ xi[2],      1, -xi[0], xi[4]],
            [-xi[1],  xi[0],      1, xi[5]],
            [     0,      0,      0,     1],
        ])
        
        o3d.visualization.draw_geometries([source, target])

    if show_results:
        images = image_pyramids[2]

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

        source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(images["j"]["color_raw"], images["j"]["depth_raw"])
        target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(images["i"]["color_raw"], images["i"]["depth_raw"])

        source = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )
        target = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )

        source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        T = np.array([
            [     1, -xi[2],  xi[1], xi[3]],
            [ xi[2],      1, -xi[0], xi[4]],
            [-xi[1],  xi[0],      1, xi[5]],
            [     0,      0,      0,     1],
        ])
        
        o3d.visualization.draw_geometries([source, target])