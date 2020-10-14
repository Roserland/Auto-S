"""
    Roserland:
    显微镜控制通信

    Waiting for Haoan Sun to finish this part
"""
import os
import time
import numpy as np
import cv2
import torch
from torchvision.models import DenseNet


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


def cell_counting(binary_img):
    """
    针对二值化图像, 计算细胞数量
    :param binary_img:
    :return:
    """
    res_num = 10

    return res_num


def micro_cell_counting(img_path, model_state_path, temp_dir='./temps/micro_imgs'):
    """
    处理显微镜图片
    :param img_path:            显微镜下细胞图片
    :param model_state_path:    深度学习算法模型路径, 用以导入模型
    :param temp_dir:            临时文件存储路径
    :return:                    细胞数量
    """
    img = cv2.imread(img_path)
    model_state_dict = torch.load(model_state_path)
    model = DenseNet()

    # loading pre-trained model_state_dicts
    result_img = model(img)
    count_num = cell_counting(binary_img=result_img)

    return count_num
