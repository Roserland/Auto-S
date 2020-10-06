"""
    显微镜控制通信

"""
import os
import time
import numpy as np
import cv2

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

    save_path = os.path.join(save_dir, img_name)

    # 图像采集

    pass