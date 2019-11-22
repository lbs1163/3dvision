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
            "grayscale": {
                "raw": grayscale_raw,
                "gaussian": grayscale_gaussian,
                "sobel": {
                    "dx": grayscale_sobel_dx,
                    "dy": grayscale_sobel_dy
                }
            },
            "depth": depth_raw
        }

    return {"i": read(file_color1, file_depth1), "j": read(file_color2, file_depth2)};

def task_1_4_calculate_jacobian_matrix(image):
    [v, u] = np.indices(image["grayscale"]["gaussian"].shape)
    d = np.reshape(image["depth"], (image["depth"].size, 1))
    h_x = False

    dx = np.reshape(image["grayscale"]["sobel"]["dx"], (image["grayscale"]["sobel"]["dx"].size, 1))
    dy = np.reshape(image["grayscale"]["sobel"]["dy"], (image["grayscale"]["sobel"]["dy"].size, 1))

    I_g = np.concatenate((dx, dy), 1)

    g_s = np.array([
        [f_x/s_z,         0, - (f_x * s_x) / (s_z ** 2)],
        [      0, f_y / s_z, - (f_y * s_y) / (s_z ** 2)],
    ])

    s_xi = np.array([
        [   0,  h_z, -h_y, 1, 0, 0],
        [-h_z,    0,  h_z, 0, 1, 0],
        [ h_y, -h_x,    0, 0, 0, 1],
    ])

    J_r = np.matmul(I_g, np.matmul(g_s, s_xi))

    return J_r

if __name__ == "__main__":
    show_results = False

    images = task_1_3_apply_filters(
        "./data/image00834.png", "./data/depth00834.png",
        "./data/image00894.png", "./data/depth00894.png"
    )

    jacobian_matrix = task_1_4_calculate_jacobian_matrix(images["i"])

    print(jacobian_matrix.shape)

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