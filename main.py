#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import optical_flow
import trajectory
from mpl_toolkits.mplot3d import Axes3D

camera_matrix = np.identity(3)
img_dir = './data/00/image_0/'
img_strings = sorted(os.listdir(img_dir))
n_frames = 200
# n_frames = len(img_strings)
calc_pos = np.zeros((n_frames, 3))
calc_ori = np.zeros((n_frames, 3, 3))

true_pos = np.zeros_like(calc_pos)
true_ori = np.zeros_like(calc_ori)

true_transforms = np.zeros((n_frames, 3, 4))

truth_file = open('./data/00/dataset/poses/00.txt').readlines()
for i, line in enumerate(truth_file[:n_frames]):
    elems = np.float32(line.split(' '))
    true_transforms[i, :, :] = np.reshape(elems, (3, 4))


def get_mag(trans_mat_diff):
    mag = np.linalg.norm(trans_mat_diff[:, -1])
    return mag


def transform_points(pos, ori, R, t):
    next_pos = pos + np.matmul(ori, t)
    next_ori = np.matmul(R, ori)
    return next_pos, next_ori


calc_ori[0, :, :] = np.identity(3)
true_ori[0, :, :] = true_transforms[0, 0:3, 0:3]
# %%

for t_step, img_string in enumerate(img_strings[:n_frames-1]):
    img1_color = cv2.imread(img_dir + img_string)
    img2_color = cv2.imread(img_dir + img_strings[t_step + 1])

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(100)
    kp1, kp2 = optical_flow.get_mutual_kps(img1, img2, fast)
    # E = trajectory.get_essential_matrix(kp1, kp2, camera_matrix)
    # R, t = trajectory.get_transformation(kp1, kp2, E, camera_matrix)

    E, mask = cv2.findEssentialMat(kp1, kp2, focal=7.18856e2, pp=(607.1928, 185.2157), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    pts, R, t, mask = cv2.recoverPose(E, kp1, kp2, focal = 7.18856e2, pp=(607.1928, 185.2157))
    # print("rotation: ", R, "translation: ", t)

    # get the true transformation matrix for each time step and its magnitude
    trans_mat = true_transforms[t_step, :, :]
    # Get magnitude for calculated transformation from the odom data
    trans_mat_diff = np.abs(true_transforms[t_step, :, :] - true_transforms[t_step+1, :, :])
    t = get_mag(trans_mat_diff)*t
    # t = [t[0], t[2], t[1]]

    curr_pos = np.reshape(calc_pos[t_step], (3, 1))
    curr_ori = calc_ori[t_step]

    true_t = np.reshape(trans_mat[:, -1], (3, 1))
    true_R = trans_mat[0:3, 0:3]

    curr_true_pos = np.reshape(true_pos[t_step], (3, 1))
    curr_true_ori = true_ori[t_step]

    # Calculated trajectory
    next_pos, next_ori = transform_points(curr_pos, curr_ori, R, t)

    # ground truth
    next_true_pos, next_true_ori = transform_points(curr_true_pos, curr_true_ori, true_R, true_t)

    calc_pos[t_step + 1] = np.reshape(next_pos, (3))
    calc_ori[t_step + 1] = next_ori

    true_pos[t_step + 1] = np.reshape(next_true_pos, (3))
    true_ori[t_step + 1] = next_true_ori

    # true_pos[t_step + 1] = np.reshape(next_true_pos, (3))
    # true_ori[t_step + 1] = next_true_ori

# plt.plot(calc_pos[:, 0], calc_pos[:, 1], 'b-')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(calc_pos[:,0], -calc_pos[:,2], -calc_pos[:,1], 'b')
# plot the true trajectory
ax.scatter(true_transforms[:,0,3], true_transforms[:,2,3], true_transforms[:,1,3], 'r')
# plt.plot(true_pos[:, 1], true_pos[:, 2], 'r-')
plt.show()

# %%
