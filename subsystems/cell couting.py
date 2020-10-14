"""
    细胞计数通信文件

    输入 显微镜下图像（from microscope）
    输出 细胞估计数目
"""
import os
import time
import numpy as np
import cv2

def is_valid_density(cell_num_per_img, threshold=1e3):
    if cell_num_per_img >= threshold:
        return True
    else:
        return False


def mscp_capture(save_dir='../temp/', img_name=None):
    """
    采集显微镜下图像;
    并将其存到固定位置，或者通过进程间通信发送给
    :return:
    """

    if img_name is not None:
        pass
    else:
        img_name = time.time()

    # TODO: capture a picture from the microscope
    save_path = os.path.join(save_dir, img_name)

    pass
