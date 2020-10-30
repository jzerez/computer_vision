#!/usr/bin/env python3

def calc_grad(img):
    ygrad = img[1:] - img[:-1]
    xgrad = (img.T[1:] - img.T[:-1]).T
    return (xgrad, ygrad)


def calc_grad_orientation(xgrad, ygrad):
    orientations = np.arctan2(ygrad, xgrad)
    return orientations
