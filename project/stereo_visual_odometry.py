#!/usr/bin/env python

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import random

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return x, y, z

def generate3DPoints(points_2d_left, points_2d_right, P0, P1):
    numPoints = points_2d_left.shape[0]
    points_3d = np.zeros((numPoints, 3))

    for i in range(numPoints):
        pLeft = points_2d_left[i,:]
        pRight = points_2d_right[i,:]

        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * P0[2,:] - P0[0,:]
        X[1,:] = pLeft[1] * P0[2,:] - P0[1,:]
        X[2,:] = pRight[0] * P1[2,:] - P1[0,:]
        X[3,:] = pRight[1] * P1[2,:] - P1[1,:]

        [u,s,v] = np.linalg.svd(X)
        v = v.transpose()
        vSmall = v[:,-1]
        vSmall /= vSmall[-1]

        points_3d[i, :] = vSmall[0:-1]

    return points_3d

def find_rotation_and_translation(matching_3d_points):
    matching_3d_points = np.array(matching_3d_points)

    points_before = np.transpose(matching_3d_points[:, 0:3])
    points_after = np.transpose(matching_3d_points[:, 3:6])

    centroid_before = np.reshape(np.average(points_before, axis=1), (3, 1))
    centroid_after = np.reshape(np.average(points_after, axis=1), (3, 1))

    H = (points_after - centroid_after) @ np.transpose(points_before - centroid_before)

    u, s, v = np.linalg.svd(H)

    R = v @ np.transpose(u)

    if np.linalg.det(R) < 0:
        u, s, v = np.linalg.svd(R)
        v[:, 2] = v[:, 2] * -1
        R = v @ np.transpose(u)
    
    t = centroid_before - R @ centroid_after

    return R, t

def is_inlier(m, matching_3d_point):
    R, t = m

    point_before = np.transpose(matching_3d_point[0:3])
    return True

def ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)

    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0

        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break

    return best_model, best_ic

if __name__ == "__main__":
    debug = True
    save_image = False
    use_stereo = True

    sequence = 0
    start_frame = 0
    end_frame = 1000

    max_iterations = 1
    sample_size = 6
    goal_inliers_ratio = 0.9

    datapath = './data/data_odometry_gray/dataset/sequences/' + '{0:02d}'.format(sequence)
    left_image_path = datapath + '/image_0/'
    right_image_path = datapath + '/image_1/'

    calib_file = open(datapath + '/calib.txt', 'r').readlines()
    projection_matrix_left = np.reshape(np.array(calib_file[0].split()[1:]), (3, 4)).astype(np.float)
    projection_matrix_right = np.reshape(np.array(calib_file[1].split()[1:]), (3, 4)).astype(np.float)
    focal_left = projection_matrix_left[0, 0]
    pp_left = (projection_matrix_left[0, 2], projection_matrix_left[1, 2])
    baseline = 0.54

    ground_truth = []
    if debug:
        ground_truth = open('./data/data_odometry_poses/dataset/poses/' + '{0:02d}'.format(sequence) + '.txt', 'r').readlines()
        ground_truth_file = open('./output/ground_truth/' + '{0:02d}'.format(sequence) + '.txt', 'w')

    estimates_file = open('./output/estimates/' + '{0:02d}'.format(sequence) + '.txt', 'w')

    orb = cv.ORB_create()
    bf = cv.BFMatcher()
    block = 11
    stereo = cv.StereoSGBM_create(minDisparity=0,numDisparities=32, blockSize=block, P1=block*block*8, P2=block*block*32)

    image_before_left = None
    image_before_right = None
    keypoints_before_left = None
    descriptors_before_left = None
    disparity_before = None

    image_after_left = None
    image_after_right = None
    keypoints_after_left = None    
    descriptors_after_left = None
    disparity_after = None

    roll_arr = np.zeros(end_frame - start_frame)
    roll_gt_arr = np.zeros(end_frame - start_frame)
    pitch_arr = np.zeros(end_frame - start_frame)
    pitch_gt_arr = np.zeros(end_frame - start_frame)
    yaw_arr = np.zeros(end_frame - start_frame)
    yaw_gt_arr = np.zeros(end_frame - start_frame)

    surge_arr = np.zeros(end_frame - start_frame)
    surge_gt_arr = np.zeros(end_frame - start_frame)
    sway_arr = np.zeros(end_frame - start_frame)
    sway_gt_arr = np.zeros(end_frame - start_frame)
    heave_arr = np.zeros(end_frame - start_frame)
    heave_gt_arr = np.zeros(end_frame - start_frame)

    roll_fig = plt.subplot(6, 1, 1)
    pitch_fig = plt.subplot(6, 1, 2)
    yaw_fig = plt.subplot(6, 1, 3)
    surge_fig = plt.subplot(6, 1, 4)
    sway_fig = plt.subplot(6, 1, 5)
    heave_fig = plt.subplot(6, 1, 6)

    roll_fig.set_ylim([-0.03, 0.03])
    pitch_fig.set_ylim([-0.1, 0.1])
    yaw_fig.set_ylim([-0.02, 0.03])
    surge_fig.set_ylim([-0.4, 0.4])
    sway_fig.set_ylim([-0.1, 0.1])
    heave_fig.set_ylim([-0.1, 1.5])

    roll_fig.set_ylabel("roll")
    pitch_fig.set_ylabel("pitch")
    yaw_fig.set_ylabel("yaw")
    surge_fig.set_ylabel("surge")
    sway_fig.set_ylabel("sway")
    heave_fig.set_ylabel("heave")

    for frame_before in range(start_frame, end_frame):

        ### load images and extract features from only left images
        
        if frame_before == start_frame:
            image_before_left = cv.imread(left_image_path + '{0:06d}'.format(frame_before) + '.png', 0)
            image_before_right = cv.imread(right_image_path + '{0:06d}'.format(frame_before) + '.png', 0)

            keypoints_before_left, descriptors_before_left = orb.detectAndCompute(image_before_left, None)
            disparity_before = np.divide(stereo.compute(image_before_left, image_before_right).astype(np.float32), 16.0)

            if save_image:
                image_before_left_with_keypoints = cv.drawKeypoints(image_before_left, keypoints_before_left, None, color=(0, 255, 0), flags=0)
                cv.imwrite('./output/orb/' + '{0:06d}'.format(frame_before) + '.png', image_before_left_with_keypoints)
                cv.imwrite('./output/disparity/' + '{0:06d}'.format(frame_before) + '.png', disparity_before)
        else:
            image_before_left = image_after_left
            image_before_right = image_after_right

            keypoints_before_left = keypoints_after_left
            descriptors_before_left = descriptors_after_left
            disparity_before = disparity_after
        
        image_after_left = cv.imread(left_image_path + '{0:06d}'.format(frame_before + 1) + '.png', 0)
        image_after_right = cv.imread(right_image_path + '{0:06d}'.format(frame_before + 1) + '.png', 0)

        keypoints_after_left, descriptors_after_left = orb.detectAndCompute(image_after_left, None)
        disparity_after = np.divide(stereo.compute(image_after_left, image_after_right).astype(np.float32), 16.0)

        if save_image:
            image_after_left_with_keypoints = cv.drawKeypoints(image_after_left, keypoints_after_left, None, color=(0, 255, 0), flags=0)
            cv.imwrite('./output/orb/' + '{0:06d}'.format(frame_before + 1) + '.png', image_after_left_with_keypoints)
            cv.imwrite('./output/disparity/' + '{0:06d}'.format(frame_before + 1) + '.png', disparity_after)

        ### match features in left images

        matches = bf.knnMatch(descriptors_before_left, descriptors_after_left, k=2)
        
        good = []
        matching_points_before_left = []
        matching_points_after_left = []

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                point_before = np.int32(keypoints_before_left[m.queryIdx].pt)
                point_after = np.int32(keypoints_after_left[m.trainIdx].pt)

                if np.linalg.norm(point_before - point_after) < 100:
                    good.append(m)
                    matching_points_after_left.append(point_after)
                    matching_points_before_left.append(point_before)
        
        matching_points_before_left = np.int32(matching_points_before_left)
        matching_points_after_left = np.int32(matching_points_after_left)
        
        if save_image:
            draw_params = dict(matchColor = (0, 255, 0),
                                singlePointColor = (255, 0, 0),
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            matching = cv.drawMatches(image_before_left, keypoints_before_left, image_after_left, keypoints_after_left, good, None, **draw_params)

            cv.imwrite('./output/matching/' + '{0:06d}'.format(frame_before) + '.png', matching)
        
        ### compute camera motion between 2 frames

        R, t = None, None
        
        if use_stereo:
            matching_points_left_before = []
            matching_points_right_before = []
            matching_points_left_after = []
            matching_points_right_after = []

            for i in range(matching_points_before_left.shape[0]):
                point_before_left = matching_points_before_left[i]
                point_after_left = matching_points_after_left[i]

                point_disparity_before = disparity_before[point_before_left[1], point_before_left[0]]
                point_disparity_after = disparity_after[point_after_left[1], point_after_left[0]]

                if point_disparity_before > 0.0 and point_disparity_before < 100.0 and point_disparity_after > 0 and point_disparity_after < 100.0:
                    point_before_right = np.copy(point_before)
                    point_before_right[0] = point_before_right[0] - point_disparity_before

                    point_after_right = np.copy(point_after)
                    point_after_right[0] = point_after_right[0] - point_disparity_before

                    matching_points_left_before.append(point_before)
                    matching_points_right_before.append(point_before_right)
                    matching_points_left_after.append(point_after)
                    matching_points_right_after.append(point_after_right)
            
            matching_points_left_before = np.array(matching_points_left_before)
            matching_points_right_before = np.array(matching_points_right_before)
            matching_points_left_after = np.array(matching_points_left_after)
            matching_points_right_after = np.array(matching_points_right_after)

            matching_points_3d_before = generate3DPoints(matching_points_left_before, matching_points_right_before, projection_matrix_left, projection_matrix_right)
            matching_points_3d_after = generate3DPoints(matching_points_left_after, matching_points_right_after, projection_matrix_left, projection_matrix_right)
            matching_points_3d = np.transpose(np.hstack((matching_points_3d_before, matching_points_3d_after)))            
                                            
            (R, t), ic = ransac(
                matching_points_3d,
                find_rotation_and_translation,
                is_inlier,
                sample_size,
                int(len(matching_points_3d) * goal_inliers_ratio),
                max_iterations
            )
            
        else:
            E, mask = cv.findEssentialMat(matching_points_after_left, matching_points_before_left, focal=focal_left, pp=pp_left, method=cv.RANSAC)

            inlier_matching_points_before_left = matching_points_before_left[mask.ravel()==1]
            inlier_matching_points_after_left = matching_points_after_left[mask.ravel()==1]

            _, R, t, mask = cv.recoverPose(E, inlier_matching_points_after_left, inlier_matching_points_before_left, focal=focal_left, pp=pp_left)

        ### calculate euler angles from rotation matrix
            
        roll, pitch, yaw = rotationMatrixToEulerAngles(R)
        surge, sway, heave = t[0, 0], t[1, 0], t[2, 0]

        estimates_file.write('{0:06e}, {1:06e}, {2:06e}, {3:06e}, {4:06e}, {5:06e}\n'.format(roll, pitch, yaw, surge, sway, heave))

        roll_arr[frame_before - start_frame] = roll
        pitch_arr[frame_before - start_frame] = pitch
        yaw_arr[frame_before - start_frame] = yaw

        surge_arr[frame_before - start_frame] = surge
        sway_arr[frame_before - start_frame] = sway
        heave_arr[frame_before - start_frame] = heave

        if debug:
            ground_truth_matrix_before = np.reshape(np.array(ground_truth[frame_before].split()), (3, 4)).astype(np.float)
            ground_truth_matrix_after = np.reshape(np.array(ground_truth[frame_before + 1].split()), (3, 4)).astype(np.float)

            rotation_before = ground_truth_matrix_before[:, 0:3]
            rotation_after = ground_truth_matrix_after[:, 0:3]

            translation_before = ground_truth_matrix_before[:, 3]
            translation_after = ground_truth_matrix_after[:, 3]

            rotation_before_inverse = np.linalg.inv(rotation_before)

            rotation = rotation_before_inverse @ rotation_after
            translation = rotation_before_inverse @ (translation_after - translation_before)

            roll_gt, pitch_gt, yaw_gt = rotationMatrixToEulerAngles(rotation)
            surge_gt, sway_gt, heave_gt = translation[0], translation[1], translation[2]

            ground_truth_file.write('{0:06e}, {1:06e}, {2:06e}, {3:06e}, {4:06e}, {5:06e}\n'.format(roll_gt, pitch_gt, yaw_gt, surge_gt, sway_gt, heave_gt))

            roll_gt_arr[frame_before - start_frame] = roll_gt
            pitch_gt_arr[frame_before - start_frame] = pitch_gt
            yaw_gt_arr[frame_before - start_frame] = yaw_gt

            surge_gt_arr[frame_before - start_frame] = surge_gt
            sway_gt_arr[frame_before - start_frame] = sway_gt
            heave_gt_arr[frame_before - start_frame] = heave_gt
            
            if frame_before % 100 == 0:

                roll_fig.plot(roll_arr[0:frame_before-start_frame+1], 'b')
                roll_fig.plot(roll_gt_arr[0:frame_before-start_frame+1], 'r')

                pitch_fig.plot(pitch_arr[0:frame_before-start_frame+1], 'b')
                pitch_fig.plot(pitch_gt_arr[0:frame_before-start_frame+1], 'r')

                yaw_fig.plot(yaw_arr[0:frame_before-start_frame+1], 'b')
                yaw_fig.plot(yaw_gt_arr[0:frame_before-start_frame+1], 'r')

                surge_fig.plot(surge_arr[0:frame_before-start_frame+1], 'b')
                surge_fig.plot(surge_gt_arr[0:frame_before-start_frame+1], 'r')

                sway_fig.plot(sway_arr[0:frame_before-start_frame+1], 'b')
                sway_fig.plot(sway_gt_arr[0:frame_before-start_frame+1], 'r')

                heave_fig.plot(heave_arr[0:frame_before-start_frame+1], 'b')
                heave_fig.plot(heave_gt_arr[0:frame_before-start_frame+1], 'r')

                plt.pause(0.001)
        
        if frame_before % 100 == 0:
            print("Frame #" + str(frame_before) + " is done");
    
    if debug:
        ground_truth_file.close()
        plt.show()

    estimates_file.close()