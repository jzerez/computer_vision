import cv2
import pdb
import numpy as np

img1 = cv2.imread('test_data/frame0000.jpg')
img2 = cv2.imread('test_data/frame0001.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
fast.setThreshold(100)
kp1 = np.array([np.array(i.pt) for i in fast.detect(img1, None)])
kp2 = np.array([np.array(i.pt) for i in fast.detect(img2, None)])

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
color = np.random.randint(0,255,(100,3))

mask = np.zeros_like(img1)
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
# draw the tracks
for i,(new,old) in enumerate(zip(good_new, good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv2.circle(img1,(a,b),5,color[i].tolist(),-1)

img = cv2.add(img1,mask)
cv2.imshow('frame',img1)
k = cv2.waitKey(30) & 0xff





a = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
print(a)
