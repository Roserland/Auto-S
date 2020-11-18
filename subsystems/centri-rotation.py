# 根据四个marker，将离心机图片旋转某个角度，从而估算出孔洞
# 同时剔除无关背景，便于识别
#

import numpy as np
import cv2
import os, yaml, time, json
import pandas as pd

def get_angle(anchor1, anchor2):
    x1, y1 = anchor1
    x2, y2 = anchor2

    diff_x = np.round(x1 - x2)
    diff_y = np.round(y1 - y2)

    sign = (diff_x - 0) * (diff_y - 0)
    sign = 1 if sign >=0 else -1

    if abs(diff_x) <= 0.0:
        angle = np.pi / 2
        return angle * sign

    k = diff_y / diff_x
    angle = np.arctan(k)
    return angle

def rotate_coords(origin_coords, alpha):
    """

    :param origin_coords: original coordinates
    :param alpha:
    :return:
    """
    m, n = origin_coords
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)

    m2 = m * sin_a + n * cos_a
    n2 = m * cos_a - n * sin_a

    return (m2, n2)


def main():
    anchor_a1 = (269.62025316455697, 50.15189873417721)
    anchor_a2 = (189.873417721519, 340.0253164556962)
    angle_a = get_angle(anchor_a2, anchor_a1)
    print('angle A is:', angle_a * 180 / np.pi)

    anchor_b1 = (375.9493670886076, 231.79746835443038)
    anchor_b2 = (81.01265822784809, 152.0506329113924)
    angle_b = get_angle(anchor_b2, anchor_b1)
    print('angle A is:', angle_b * 180 / np.pi)

    print("angle diff:", (angle_a - angle_b) * 180 / np.pi)



if __name__ == '__main__':
    main()