#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

def task_1_3_apply_filters(file_color1, file_depth1, file_color2, file_depth2):
    def read(color, depth):
        color_raw = o3d.io.read_image(file_color1)
        depth_raw = o3d.io.read_image(file_depth1)

        color_gaussian = cv.GaussianBlur(np.asarray(color_raw), (3, 3), 0)

        color_sobel_dx = cv.Sobel(np.asarray(color_raw), -1, 1, 0)
        color_sobel_dy = cv.Sobel(np.asarray(color_raw), -1, 0, 1)

        return {
            "color": {
                "raw": color_raw,
                "gaussian": color_gaussian,
                "sobel": {
                    "dx": color_sobel_dx,
                    "dy": color_sobel_dy
                }
            },
            "depth": depth_raw
        }

    return [read(file_color1, file_depth1), read(file_color2, file_depth2)];

if __name__ == "__main__":
    images = task_1_3_apply_filters(
        "./data/image00834.png", "./data/depth00834.png",
        "./data/image00894.png", "./data/depth00894.png"
    )

    plt.subplot(2, 3, 1)
    plt.imshow(images[0]["color"]["raw"])
    plt.subplot(2, 3, 2)
    plt.imshow(images[0]["color"]["gaussian"])
    plt.subplot(2, 3, 3)
    plt.imshow(images[0]["depth"])
    plt.subplot(2, 3, 5)
    plt.imshow(images[0]["color"]["sobel"]["dx"])
    plt.subplot(2, 3, 6)
    plt.imshow(images[0]["color"]["sobel"]["dy"])
    plt.show()

    plt.subplot(2, 3, 1)
    plt.imshow(images[1]["color"]["raw"])
    plt.subplot(2, 3, 2)
    plt.imshow(images[1]["color"]["gaussian"])
    plt.subplot(2, 3, 3)
    plt.imshow(images[1]["depth"])
    plt.subplot(2, 3, 5)
    plt.imshow(images[1]["color"]["sobel"]["dx"])
    plt.subplot(2, 3, 6)
    plt.imshow(images[1]["color"]["sobel"]["dy"])
    plt.show()