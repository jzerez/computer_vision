#!/usr/bin/env python3
#%%
import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os


def get_keypoints(img, fast=None, threshold=100):
    """
    Uses cv2's built in FAST feature detector to find keypoints in an image

    Parameters:
        img (np.array): grayscale image to find keypoints in
        fast (FastFeatureDetector): FAST object from cv2 (optional)
        threshold (int): threshold value for FAST object
    Returns:
        kps (np.array): a (nx1x2) float32 array containing keypoints in image coordinates

    """
    if not fast:
        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(threshold)
    kps = fast.detect(img1, None)
    return kps

def calc_next_kps(img1, img2, kp1s):
    """
    Uses cv2's built in Optical Flow function to find corresponding keypoints between 2 images
    
    Parameters:
        img1 (np.array): grayscale image for first frame
        img2 (np.array): grayscale image for second frame
        kp1s (np.array): a (nx1x2) float32 array for keypoints in the first frame
    Returns:
        valid_old_kp (np.array): keypoints in the first image with correspondences in the second image
        valid_new_kp (np.array): keypoints in the second image with correspondences in the first image
    """

    kp1_points = np.array([[i.pt] for i in kp1s], dtype=np.float32)
    kp2_points, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, kp1_points, None)
    
    valid_new_kp = kp2_points[st == 1]
    valid_old_kp = kp1_points[st == 1]

    return valid_old_kp, valid_new_kp

def get_kps(img1, img2, fast=None, threshold=100):
    """
    Gets the mutual and corresponding keypoints from two frames

    Parameters: 
        img1 (np.array): grayscale image for first frame
        img2 (np.array): grayscale image for second frame
        fast (FastFeatureDetector): FAST object from cv2 (optional)
        threshold (int): threshold value for FAST object
    Returns:
        valid_old_kp (np.array): keypoints in the first image with correspondences in the second image
        valid_new_kp (np.array): keypoints in the second image with correspondences in the first image
    """
    initial_kp1 = get_keypoints(img1, fast=fast, threshold=threshold)
    return calc_next_kps(img1, img2, initial_kp1)

#%%
def approx_trajectory(path):
    """
    Iterates through images and approximates trajectory of camera in 3D

    Parameters:
        path (String): the path of a folder containing the images

    Returns:
        trajectory (tuple(np.array, np.array)): the trajectory of the camera in an inertial reference frame
    """
    filenames = []
    for (roots, dirs, files) in os.walk(path, topdown=True):
        filenames.append(files)
        

    filenames = sorted(filenames[0], key = str.lower)
    # print(filenames)
    

    for i in range(len(filenames)-1):
        imgName1 = filenames[i]
        imgName2 = filenames[i+1]

        img1 = cv2.imread(f'{path}/{imgName1}')
        print(f'{path}/{imgName1}')
        img2 = cv2.imread(f'{path}/{imgName2}')

        # img1 = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)

        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(100)

        kp1 = get_keypoints(img1, fast)
        valid_kp1, valid_kp2 = calc_next_kps(img1, img2, kp1)

#%%


if __name__ == "__main__":

# %%    
    approx_trajectory("/home/james/catkin_ws/src/comprobo20/computer_vision/data/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_00/data")
# %%
    approx_trajectory
    # img1c = cv2.imread('test_data/frame0000.jpg')
    # img2c = cv2.imread('test_data/frame0001.jpg')
    # img1 = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)

    # fast = cv2.FastFeatureDetector_create()
    # fast.setThreshold(100)

    # kp1 = get_keypoints(img1, fast)
    # valid_kp1, valid_kp2 = calc_next_kps(img1, img2, kp1)

    # color = np.random.randint(0,255,(100,3))
    # mask = np.zeros_like(img1c)
    # stacked_img = cv2.addWeighted(img1c, 0.5, img2c, 0.5, 0)

    # print(valid_kp1)
    # for i,(new,old) in enumerate(zip(valid_kp1, valid_kp2)):
    #     a,b = new.astype('int').ravel()
    #     c,d = old.astype('int').ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(stacked_img,(a,b),5,color[i].tolist(),-1)

    # E, mask = cv2.findEssentialMat(valid_kp1, valid_kp2, focal = 1.0, pp=(0.,0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # pts, R, t, mask = cv2.recoverPose(E, valid_kp1, valid_kp2)
    # print("rotation: ", R, "translation: ", t)
    
    # TODO(1): create a loop to go through each photo
    # TODO(2): determine the scale of each translation. They are returned in unit vectors
    # TODO(3): integrate up the subsequent rotations and translations to plot a path 
    # TODO(4): plot the path in 3D 

    

    # img = cv2.add(frame,mask)
    # cv2.imshow('frame',img), plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # # Initiate FAST object with default values
    # fast = cv2.FastFeatureDetector_create()
    # fast.setThreshold(100)
    # kp1 = np.array([np.array(i.pt, dtype=np.float32) for i in fast.detect(img1, None)])
    # kp2 = np.array([np.array(i.pt, dtype=np.float32) for i in fast.detect(img2, None)])
    # kp1 = np.reshape(kp1, (kp1.shape[0], 1, 2))


    # feature_params = dict( maxCorners = 100,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )

    # p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    # color = np.random.randint(0,255,(100,3))

    # mask = np.zeros_like(img1c)
    # # calculate optical flow
    # p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None)
    # # Select good points
    # good_new = p1[st==1]
    # good_old = kp1[st==1]
    # # draw the tracks
    # stacked_img = cv2.addWeighted(img1c, 0.5, img2c, 0.5, 0)
    # for i,(new,old) in enumerate(zip(good_new, good_old)):
    #     a,b = new.astype('int').ravel()
    #     c,d = old.astype('int').ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(stacked_img,(a,b),5,color[i].tolist(),-1)


    # img = cv2.add(frame,mask)
    # cv2.imshow('frame',img), plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
