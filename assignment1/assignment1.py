#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d

def task_1_3_extract_and_match_features(img1, img2):
    orb = cv.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    img1_w_kps = cv.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
    img2_w_kps = cv.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

    cv.imwrite('./output/image1_orb.png', img1_w_kps)
    cv.imwrite('./output/image2_orb.png', img2_w_kps)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = (255, 0, 0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img3 = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)

    cv.imwrite('./output/orb_matching.png', img3)

    return matches, keypoints1, keypoints2

def task_1_4_compute_fundamental_matrix(matches, kp1, kp2, img1, img2):
    good = []
    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    def drawlines(img1, img2, lines, pts1, pts2):
        h, w = img1.shape
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            (x0, y0) = map(int, [0, -r[2]/r[1]])
            (x1, y1) = map(int, [w, -(r[2]+r[0]*w)/r[1]])
            img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
        
        return img1, img2
    
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    cv.imwrite('./output/epipolar_lines_1.png', img5)
    cv.imwrite('./output/epipolar_lines_2.png', img3)
    
    return F, img5, img3

def task_1_5_compute_essential_matrix(F, K):
    return np.matmul(np.matmul(np.transpose(K), F), K)

def task_1_6_get_camera_poses(E):
    U, S, Vt = np.linalg.svd(E, full_matrices=True)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    R1 = np.matmul(np.matmul(U, W), Vt)
    R2 = np.matmul(np.matmul(U, np.transpose(W)), Vt)
    u3 = U[:, 2]

    return R1, R2, u3

def task_2_1_rectify_images(K, R, t, distCoeffs, img1, img2):
    rectify_params = dict(alpha = 1.0, flags=0)
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K, distCoeffs, K, distCoeffs, (img1.shape[1], img1.shape[0]), R, t, **rectify_params)

    map1, map2 = cv.initUndistortRectifyMap(K, distCoeffs, R1, P1, (img1.shape[1], img1.shape[0]), cv.CV_32FC1)
    map3, map4 = cv.initUndistortRectifyMap(K, distCoeffs, R2, P2, (img2.shape[1], img2.shape[0]), cv.CV_32FC1)

    rect1 = cv.remap(img1, map1, map2, cv.INTER_NEAREST)
    rect2 = cv.remap(img2, map3, map4, cv.INTER_NEAREST)

    cv.imwrite('./output/rectified_1.png', rect1)
    cv.imwrite('./output/rectified_2.png', rect2)

    return rect1, rect2

if __name__ == "__main__":
    K = np.array([[1112.60959, 0.0, 724.315664],
                  [0.0, 1122.20828, 520.183459],
                  [0.0, 0.0, 1.0]])
    distCoeffs = np.transpose([0.28864512, -1.51224766, -0.00702437, 0.00289229, 2.43952476])

    img1 = cv.imread('./data/lab1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('./data/lab2.jpg', cv.IMREAD_GRAYSCALE)

    matches, kp1, kp2 = task_1_3_extract_and_match_features(img1, img2)
    F, epipolar_lines_1, epipolar_lines_2 = task_1_4_compute_fundamental_matrix(matches, kp1, kp2, img1, img2)
    E = task_1_5_compute_essential_matrix(F, K)
    R1, R2, t = task_1_6_get_camera_poses(E)
    rect1, rect2 = task_2_1_rectify_images(K, R1, t, distCoeffs, epipolar_lines_1, epipolar_lines_2)