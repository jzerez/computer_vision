# computer_vision
*James Ho and Jonathan Zerez*

## Project Intro

There are many ways for a mobile robot to know where it is in space. One could use GPS, wheel encoders, IMUs, or a combination of all or more sensors. One approach is to use visual information to localize the robot. The goal for our project is to implement a visual odometry algorithm in Python to plot the path of a vehicle. We used the the KITTI dataset to obtain undistorted camera data and ground truth paths and OpenCV for the common computer vision functions. We ultimately wanted an implementation of a visual odometry algorithm that would produce relatively accurate results.

### Methodology
How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.

Reconstructing the path of a camera from a sequence of images requires two main steps. Here is a high level overview of what goes into creating visual odometry:
1. Find the key-points in each frame
2. Label each key-point based on the image intensity gradient in its vicinity
3. Find the correspondence between key-points in consecutive frames
4. Use RANSAC to calculate the best Essential Matrix, which transforms key-points in one frame to the next
5. Decompose the Essential Matrix into a translation vector and rotation matrix
6. Apply these transformations to the previous pose to get the new pose

#### Finding key-points
To find the key-points, we use `cv2`'s implementation of a FAST corner detector. For a given pixel *p*, it looks at the circle *c* of 16 pixels that are a radius of 3 away from *p*. If there is a continuous group of *N* pixels in *c* that are all lighter than *p* plus a threshold value, *t*, or if there is a continuous group of *N* pixels in *c* that are all darker than *p* minus *t*, then *p* is labeled as a corner.

![fast](./assets/FAST.PNG)

The figure above is from the [official site for the FAST algorithm](http://www.edwardrosten.com/work/fast.html). It illustrates the 16 pixels around *p*, *c*, and illustrates the continuous group of pixels in *c* that are all brighter than *p* plus some threshold value *t*.

Accomplishing this in python is really easy thanks to openCV.

```
fast = cv2.FastFeatureDetector_create()
fast.setThreshold(100)
kps = fast.detect(img, None)
```

#### Labeling key-points
In order to label key-points we used the Lucas-Kanade (LK) method in order to calculate the image flow vector between subsequent frames. From the [Wikipedia article for the LK method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method):

*\"The Lucas–Kanade method assumes that the displacement of the image contents between two nearby instants (frames) is small and approximately constant within a neighborhood of the point p under consideration"*

This means that first order tailor expansion for the gradient in the neighborhood around a target pixel is sufficient to capture changes from one frame to the next. In other words, the element-wise product between the image flow vector *V*, and the partial derivatives of the image in the x and y direction *I* around a pixel *p*, is equal to the time derivative of the image around the pixel *p*.

<img src="https://latex.codecogs.com/gif.latex?I_x(p)V_x+I_y(p)V_y=-I_t(p)" />

Because we consider a neighborhood 3x3 of pixels around each key-point, there are more equations than unknowns and the problem is over constrained. The LK method uses the least squares principle to create an estimate for *V* that minimizes the sum of the squared error of each of the terms corresponding to the neighborhood of pixels that are considered.

Accomplishing this in openCV is very straightforward:
```
kp2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None)
```
Here, `kp1` refers to the key-points found in the first image, `img1`, and `kp2` refers to the corresponding key-points found in the second image, `img2`. OpenCV automatically determines whether a key-point from the original image contains a corresponding keypoint in the second image.

`cv2.calcOpticalFlowPyrLK()` finds the common keypoints between two frames. In subsequent frames, certain keypoints may lose correspondence as the original keypoints eventually leave the frame. In order to ensure that there are a sufficient number of keypoints, we create a keypoint threshold which is 80% of the original number of keypoints calculated. If we drop below 80%, the program automatically resamples keypoints. Below is a graph that shows the keypoints used during the calculation. You can observe declines followed by sharp peaks. This represents the number of  keypoints dwindling until they are suddently resampled.

![num_kps](./num_kps_v2.png)

#### Calculating the Essential Matrix
The essential matrix, *E* is defined as a transformation from one set of key-points to another, like so:

<img src="https://latex.codecogs.com/gif.latex?k_1^TEk_2=0" />

where *k1* corresponds to the set of key-points in the first frame, and *k2* corresponds to the set of key-points in the second frame. Calculating this matrix can be done by sampling five points, as the essential matrix *E* has 5 unknowns. Because we have many more than five key-points, we use RANSAC, or Random Sampling Consensus, to iteratively select sets of random five key-points, compute the corresponding essential matrix, and see how well that essential matrix is able to transform the rest of the key-points from the first frame to the second frame.

Samples of five key-points are continually selected until an essential matrix *E* is found with a sufficient number of inliers (or until a specified iteration count is exceeded).

Calculating *E* in openCV is also easy:
```
E = cv2.findEssentialMat(kp2, kp1, focal, pp, method=cv2.RANSAC, prob=0.999, threshold=1.0);
```

#### Updating the pose
The essential matrix can be decomposed into a translation vector of unit length (scale is unknown) *t*, and a rotation matrix, *R*. This is accomplished using singular value decomposition, which expresses the original matrix *E* as combination of rotational and scaling transformations.

Once *R* and *t* are calculated, updating the previous pose to reflect the new transformation that has been calculated is done by the following:

<img src="https://latex.codecogs.com/gif.latex?R_{pos}=RR_{pos}" />.

<img src="https://latex.codecogs.com/gif.latex?t_{pos}=t_{pos}+tR_{pos}" />

Where <img src="https://latex.codecogs.com/gif.latex?t_{pos}" /> is the current position of the camera, and <img src="https://latex.codecogs.com/gif.latex?R_{pos}" /> is the current rotational orientation of the camera.


Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.


One major design decision was using OpenCV to do a lot of the heavy lifting. We wanted to have a functional algorithm in a relatively short timeframe, so instead of building each component of the algorithm, we decied to us an open sourced library.

Another design decision was picking what keypoint tracker to use. We looked to literature and test result to determine which tracking algorithm is the fastest. According to [this test data](https://computer-vision-talks.com/2011-07-13-comparison-of-the-opencv-feature-detection-algorithms/), we found that the FAST algorithm was able to detect the most quickly out of the other options. A quick detector enables us to track the path of a mobile robot in real time with minimal delay.

#### Finding Optimal Keypoint Threshold
We perfomed a parameter sweep to determine what the optimal value for the keypoint threshold is. However, we only computed the value based on the first 500 frames. As a result, when we went through the entire dataset using this value, we ended up with a very large error. Given the time it takes for the parameter sweep and the algorithm to run, we ended up sticking to a the arbitarty value of 100. At the time of this writeup, we are currently running the whole parameter sweep out of curiosity. From what we can observe, it seems like around a threshold of 40-45 is pretty good. 
![results](./errorVsThreshold.png)

### Results
All in all, we were able to produce the following plot, where the orange line represents the ground truth, and the blue line represents the calculated trajectory of the car based on visual odometry.

![results](./full_course_v2.png)

It certainly tends to drift over time, but because we only have access to one stream of information, we have no way of accounting for drift over time. We can look at a brief snapshot at the beginning of the video where the impacts of drift are minimal.

![short_results](./500_tsteps.png)



### Challenges
Even though this project leaned very heavily on built-in openCV functions, it was difficult to find the correct way to call and implement certain functions, especially because openCV's documentation isn't always the greatest. There was definitely a good deal of trial and error in order to figure out parameters and odd quirks in the data.

Dealing with git and version control also was kind of a pain for this project. Neither of us are git-wizards in any sense, and so there were definitely some rather jank solutions to git problems (like `git reset --hard`, or just recloning the entire repo). We made a bunch of branches, but we probably wouldn't have needed to if we were good at git, but the extra branches were certainly helpful.  

What would you do to improve your project if you had more time?

The main thing we would improve given more time would be the accuracy of the odometry. A visual comparison between our path and the ground truth shows that errors greatly accumulate over the course of the video. This led to a large difference between the ground truth and our computed path. One improvement would be to minimize the lateral movement of the vehicle when performing transformations. This is because we know the camera is mounted on a car and it would be extremely unlikely for it to translate sideways without moving forwards.

Another improvement is to use the IMU and our visual odometry to approximate the path. This can be done using a kalman filter, something we would have looked into.

Another approach is to research and implement a better performing algotithm entirely.

### Key Takeaways
We found that both asynchronous work time, as well as synchronous pair programming worked well for us. When doing asynchronous work, it helped to create additional branches on git in order avoid merge conflicts and avoid having to push work that was potentially unfinished or buggy and corrupt the otherwise clean main branch.
