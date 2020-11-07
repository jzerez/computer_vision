#!/usr/bin/env python3
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_essential_matrix(keypoints1, keypoints2, cameraMatrix):
    '''
    keypoints1: array of keypoints from the first image 
    keypoints2: array of keypoints from the second image
    cameraMatrix: a camera matrix for a pinhole camera model

    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    returns the essential matrix for two sets of corresponding points
    retval, mask	=	cv.findEssentialMat(	points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]	)
    '''
    # find the essential matrix
    E = cv2.findEssentialMat(keypoints1, keypoints2, cameraMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E
# %%
def get_transformation(keypoints1, keypoints2, E, cameraMatrix):
    retval, R, t, mask = cv2.recoverPose(E, keypoints1, keypoints2, cameraMatrix)
    return R, t
