import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import optical_flow
import trajectory
from mpl_toolkits.mplot3d import Axes3D


class VisualOdom:
    def __init__(self, n_frames=200, fast_threshold=42):
        self.img_dir = './data/00/image_0/'
        self.img_strings = sorted(os.listdir(self.img_dir))
        if n_frames is None:
            self.n_frames = len(self.img_strings)
        else:
            self.n_frames = n_frames
        n_frames = len(self.img_strings)

        self.true_transforms = np.zeros((n_frames, 3, 4))

        with open('./data/00/dataset/poses/00.txt') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:self.n_frames]):
                elems = np.float32(line.split(' '))
                self.true_transforms[i, :, :] = np.reshape(elems, (3, 4))
        
        self.fast = cv2.FastFeatureDetector_create()
        self.fast.setThreshold(fast_threshold)
        self.kp_threshold = -1

        self.calc_pos = np.zeros((self.n_frames, 3))
        self.calc_ori = np.zeros((self.n_frames, 3, 3))
        self.calc_ori[0, :, :] = np.identity(3)

        self.curr_pos = self.true_transforms[0, :, -1]
        self.curr_ori = self.true_transforms[0, 0:3, 0:3]

        self.kp1 = None
        self.kp2 = None
        self.n_kps = []

    def get_mag(self, t_step):
        trans_mat_diff = np.abs(self.true_transforms[t_step, :, :] - self.true_transforms[t_step+1, :, :])
        mag = np.linalg.norm(trans_mat_diff[:, -1])
        return mag

    def transform_points(self, pos, ori, R, t):
        pos = np.reshape(pos, (3,1))
        next_pos = pos + np.matmul(ori, t)
        next_ori = np.matmul(R, ori)
        return next_pos, next_ori

    def update(self, img1, img2, t_step):
        if self.kp1 is None or len(self.kp1) < self.kp_threshold:
            # resample both kp1 and kp2
            self.kp1, self.kp2 = optical_flow.get_mutual_kps(img1, img2, self.fast)
            self.kp_threshold = int(len(self.kp1) * 0.8)
        else:
            # calculate kp2 based on kp1
            self.kp1, self.kp2 = optical_flow.calc_next_kps(img1, img2, self.kp1)
        self.n_kps.append(len(self.kp1))
        # self.kp1, self.kp2 = optical_flow.get_mutual_kps(img1, img2, self.fast)
        E, mask = cv2.findEssentialMat(self.kp1, self.kp2, focal=7.18856e2, pp=(607.1928, 185.2157), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        pts, R, t, mask = cv2.recoverPose(E, self.kp1, self.kp2, focal = 7.18856e2, pp=(607.1928, 185.2157))

        # Get magnitude for calculated transformation from the odom data
        mag = self.get_mag(t_step)
        
        # if mag > 0:
        t = mag*t
        # Calculated trajectory


        next_pos, next_ori = self.transform_points(self.curr_pos, self.curr_ori, R, t)
        # else:
        #     next_pos, next_ori = self.curr_pos, self.curr_ori
        self.calc_pos[t_step + 1] = np.reshape(next_pos, (3))
        self.calc_ori[t_step + 1] = next_ori
        self.curr_pos = next_pos
        self.curr_ori = next_ori
        self.kp1 = self.kp2

    def run(self):
        for t_step in range(0, self.n_frames-1):
            img1_color = cv2.imread(self.img_dir + self.img_strings[t_step])
            img2_color = cv2.imread(self.img_dir + self.img_strings[t_step + 1])

            img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

            self.update(img1, img2, t_step)

    def plot_results(self):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        plt.plot(self.calc_pos[:,0], -self.calc_pos[:,2])
        # plot the true trajectory
        plt.plot(self.true_transforms[:,0,3], self.true_transforms[:,2,3])
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')
        plt.title('Visual Odometry vs. Ground Truth (all frames)')
        plt.legend(['Calculated pose', 'Ground Truth'])
        plt.figure()
        plt.plot(self.n_kps)
        plt.show()

    def calc_error(self):
        comp_traj = self.calc_pos[:, (0,2)] * np.array([1 , -1])
        ground_truth = self.true_transforms[:self.n_frames, (0,2), -1]
        error = np.sum(np.sqrt((ground_truth-comp_traj)**2))
        return error


        


if __name__ == "__main__":
    vo = VisualOdom(n_frames=None, fast_threshold=50)
    vo.run()
    print(vo.calc_error())
    vo.plot_results()

    # def find_best_params():
    #     '''
    #     1. Start at first threshold value
    #     2. run 500 frames
    #     3. compute error
    #     4. change param
    #     5. repeat
    #     '''
    #     thresholds = np.linspace(0, 90, 10, dtype=np.int)
    #     errors = []
    #     for i in thresholds:
    #         vo = VisualOdom(n_frames=None, fast_threshold=i)
    #         vo.run()
    #         error = vo.calc_error()
    #         errors.append(error)
    #     plt.figure()
    #     plt.plot(thresholds, errors)
    #     plt.xlabel("FAST threshold value")
    #     plt.ylabel("Total Root Mean Square Error")
    #     plt.show()

    # find_best_params()