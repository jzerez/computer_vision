# computer_vision 
*James Ho and Jonathan Zerez*

## Project Proposal
What is the main idea of your project?
* We aim to develop a visual odometry algorithm that is able to trace the path of a camera in 3D. A stretch goal would be to run this algorithm on the Neato in a 3D world and compare our odometry results to the true trajectory. 

What are your learning goals for this project?
* We are aiming to get familiar with using the various built in features of OpenCV in python. Additionally, we want to learn more about the general techniques and algorithms used for visual odometry. 

What algorithms or computer vision areas will you be exploring?
* We will be following a framework we found [here](http://avisingh599.github.io/vision/monocular-vo/). First, we need to find points of interest in the current frame by detecting corners and other easily distinguishable features. We will probably use SIFT or FAST for this. Then, we need to calculate the transform between the points of interest in one frame and the corresponding points of interest in the next frame. In the framework we've found, this is done by using RANSAC to find a strong candidate for the essential matrix, and using SVD to generate rotation and translation transformations. 

What components of the algorithm will you implement yourself, which will you use built-in code for? Why?
* We will probably start off the project by relying quite heavily on built-in openCV algorithms, and slowly creating our own replacements once we are confident in the overall behavior of the code. Creating components of the SIFT/FAST algorithm (like using kernels to select points of interest) would probably be the easiest and first thing that we'd develop on our own. 

What is your MVP?
* Implement our own visual odometry algorithm in python and test our code using the [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset. Compare to ground truth.

What is a stretch goal?
* Run the algorithm on the Neato with its simulated camera and compare its results to the ground truth data.

What do you view as the biggest risks to you being successful (where success means achieving your learning goals) on this project?
* The largest risk is not fully understanding the algorithm before we implement it. Understanding each step will allow us to debug and optimize the algorithm more quickly later on.

What might you need from the teaching team for you to be successful on this project?
* Help explaining concepts or steps in the algorithm if we cannot figure it out on our own.


## Writeup
What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.
(james)

How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.
(jon)

Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.
(james)

What if any challenges did you face along the way?
(jon)

What would you do to improve your project if you had more time?
(james)

Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.
(jon)



