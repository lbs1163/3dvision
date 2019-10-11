#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d

from common import splitfn

def task_1_3_feature_extraction(filename1, filename2):
    _, name1, _ = splitfn(filename1)
    _, name2, _ = splitfn(filename2)

    img1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    img1_w_kps = cv.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
    img2_w_kps = cv.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

    cv.imwrite('./output/' + name1 + '_orb.png', img1_w_kps)
    cv.imwrite('./output/' + name2 + '_orb.png', img2_w_kps)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite('./output/' + name1 + '_' + name2 + '_orb_matching.png', img3)

    return 'asdf'

if __name__ == "__main__":
    features = task_1_3_feature_extraction('./data/lab1.jpg', './data/lab2.jpg')