import matplotlib.pyplot as plt
import numpy as np
import cv2

#!/usr/bin/env python3


# Capture images: It, It+1,
# Undistort the above images.
# Use FAST algorithm to detect features in It, and track those features to It+1. A new detection is triggered if the number of features drop below a certain threshold.

# Use Nister’s 5-point alogirthm with RANSAC to compute the essential matrix.
# Estimate R,t from the essential matrix that was computed in the previous step.
# Take scale information from some external source (like a speedometer), and concatenate the translation vectors, and rotation matrices.

# get image
img1 = cv2.imread('test_data/frame0000.jpg')
img2 = cv2.imread('test_data/frame0001.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
fast.setThreshold(10)
# find and draw the keypoints
kp = fast.detect(img1, None)
img2 = cv2.drawKeypoints(img1, kp, None, color=(255, 0, 0))

fast.setThreshold(100)
# find and draw the keypoints
kp2 = fast.detect(img1, None)
img2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

# calculate the descriptors from the keypoints


cv2.imshow('', img2), plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
