#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

if __name__ == "__main__":
    sequence = 0
    start_frame = 0
    end_frame = 1000

    datapath = './Data/' + '{0:02d}'.format(sequence)
    left_image_path = datapath + '/image_0/'
    right_image_path = datapath + '/image_1/'

    for frame_before in range(start_frame, end_frame):
        if frame_before == 0:
            image_before_left = cv.imread(left_image_path + '{0:06d}'.format(frame_before))
            image_before_right = cv.imread(right_image_path + '{0:06d}'.format(frame_before))
        else:
            image_before_left = image_after_left
            image_before_right = image_after_right
        
        image_after_left = cv.imread(left_image_path + '{0:06d}'.format(frame_before + 1))
        image_after_right = cv.imread(right_image_path + '{0:06d}'.format(frame_before + 1))