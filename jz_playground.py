#!/usr/bin/env python3
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import optical_flow
import trajectory

camera_matrix = np.identity(3)
img_dir = './data/00/image_0/'
img_strings = sorted(os.listdir(img_dir))
poses = np.zeros((len(img_strings), 2))

for t_step, img_string in enumerate(img_strings[:50]):
    img1_color = cv2.imread(img_dir + img_string)
    img2_color = cv2.imread(img_dir + img_strings[t_step + 1])

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(100)
    kp1, kp2 = optical_flow.get_mutual_kps(img1, img2, fast)
    E = trajectory.get_essential_matrix(kp1, kp2, camera_matrix)
    R, t = trajectory.get_transformation(kp1, kp2, E, camera_matrix)
    
    curr_pose = poses[t_step]
    next_post = (R * t) + curr_pose
    poses[t_step + 1] = next_post

plt.scatter(poses[:, 1], poses[:, 2])
plt.show()

# %%
