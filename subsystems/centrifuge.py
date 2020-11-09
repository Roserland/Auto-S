"""
    输入 离心机内试管图片
    输出 相应试管所在到坐标
        可能的可以安插试管的孔洞

    最好不要利用深度图像

    排除中心点附近的黑色区域, 也会是个难得处理地方

    利用real sense摄像头, 俯视角度拍摄
    观察检测出的孔洞效果
"""

import numpy as np
import cv2
import os, yaml
import pandas as pd
from subsystems.calibration_realsense import CameraCalibrator
from subsystems.functionFiles import find_max_contours, find_min_rec, draw_rectangle


# 当前可用的 无试管插入           图片编号为   00024-00028
# 当前可用的 相同(绿色)rotor的    图片编号为   00029-00037, 27,28为纯绿色rotor
# 当前可用的 较复杂场景下          图片编号为   00018-00023

class Centrifuge(object):
    def __init__(self, color_img_path, depth_npy_path,
                 crop_center=(203, 222), crop_size=(406, 406),
                 tubes_num=4, holes_num=4,
                 tubes_bgr_thres=None, holes_bgr_thres=None, temp_img_dir='../temps/centri_temp_imgs/',
                 tubes_min_pixels=100, tubes_max_pixels=700, hole_min_pixels=200, holes_max_pixels=550):
        img = cv2.imread(color_img_path)
        depth = np.load(depth_npy_path)

        self.color_img = img
        self.depth_img = depth
        self.crop_center = crop_center
        self.crop_size = crop_size
        self.aim_tubes_num = tubes_num
        self.aim_holes_num = holes_num

        if tubes_bgr_thres == None:
            self.tubes_bgr_thres = [(0, 50), (40, 150), (85, 210)]
        else:
            self.tubes_bgr_thres = tubes_bgr_thres

        if holes_bgr_thres == None:
            self.holes_bgr_thres = [(0, 10), (0, 16), (0, 10)]
        else:
            self.holes_bgr_thres = holes_bgr_thres

        if not os.path.exists(temp_img_dir):
            os.makedirs(temp_img_dir)
        self.temp_img_dir = temp_img_dir

        self.tubes_min_pixels = tubes_min_pixels
        self.tubes_max_pixels = tubes_max_pixels
        self.hole_min_pixels = hole_min_pixels
        self.holes_max_pixels = holes_max_pixels

    def crop_img(self, crop_center, crop_size):
        """
            根据所以的图片像素中心点，和 所需要的大小框，裁剪图片
            :param crop_center:
            :param crop_size:
            :return:
            """
        w, h, c = self.color_img.shape
        _h, _w = crop_size

        assert w > _w
        assert h > _h

        half_h = _h // 2
        half_w = _w // 2
        center_y, center_x = crop_center

        upper_x = center_x - half_w
        upper_y = center_y - half_h
        if upper_x < 0 or upper_y < 0:
            print("Center Position is not Valid, please check it!")
            raise ValueError

        res = self.color_img[upper_x: upper_x + _w,
              upper_y: upper_y + _h]
        tailored_img_save_path = os.path.join(self.temp_img_dir, './tailored_img.jpg')
        cv2.imwrite(tailored_img_save_path, res)
        return res

    def transfer_to_gray(self, ):
        """彩色图像转化为灰度图像"""
        return cv2.cvtColor(src=self.color_img, code=cv2.COLOR_BGR2GRAY)

    def thres_binary_for_tubes(self, img):
        bgr_thres = self.tubes_bgr_thres
        b_thres, g_thres, r_thres = bgr_thres
        b_low, b_high = b_thres
        g_low, g_high = g_thres
        r_low, r_high = r_thres

        b, g, r = cv2.split(img)

        a_r = (r >= r_low) & (r <= r_high)
        a_g = (g >= g_low) & (g <= g_high)
        a_b = (b >= b_low) & (b <= b_high)

        thres = (a_r * a_g * a_b) * 255

        return thres.astype(np.uint8)

    def thres_binary_for_holes(self, img):
        bgr_thres = self.holes_bgr_thres
        b_thres, g_thres, r_thres = bgr_thres
        b_low, b_high = b_thres
        g_low, g_high = g_thres
        r_low, r_high = r_thres

        b, g, r = cv2.split(img)

        a_r = (r >= r_low) & (r <= r_high)
        a_g = (g >= g_low) & (g <= g_high)
        a_b = (b >= b_low) & (b <= b_high)

        thres = (a_r * a_g * a_b) * 255

        return thres.astype(np.uint8)

    def coords_transformer(self, points, center_pos=None, tailored_size=None):
        """
        由于处理的图片是裁剪后的图片, 在传递给其他函数时，需要将坐标转换
        :param points:
        :param center_pos:
        :param tailored_size:
        :return:
        """
        if center_pos == None:
            center_pos = self.crop_center
        if tailored_size == None:
            tailored_size = self.crop_size

        w, h = tailored_size
        t_cen = (w / 2, h / 2)
        x = points[0] - t_cen[0] + center_pos[0]
        y = points[1] - t_cen[1] + center_pos[1]
        return (x, y)

    def draw_rectangles(self, rect_points_list, img_name='__with_multi_rectangles.jpg'):
        src_img = self.color_img[:]
        save_path = os.path.join(self.temp_img_dir, img_name)

        length = len(rect_points_list)

        # draw rectangles
        for i in range(length):
            (pt1, pt2) = rect_points_list[i]

            cv2.rectangle(img=src_img, pt1=pt1, pt2=pt2, color=(183, 152, 63), thickness=2)
        s = cv2.imwrite(save_path, src_img)
        if not s:
            print('Not saved')
            print("Please check if the path is right or if the directory exists")
            raise ValueError

    def find_possible_areas(self, bi_img, nums=4, min_area_pixels=200, max_area_pixels=1150):
        _bi_img = bi_img[:, :]
        print(_bi_img.shape)

        contours, hierarchy = cv2.findContours(_bi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("contours num is {}".format(len(contours)))
        area = []

        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        area = np.array(area)

        if len(area) == 0:
            print('***************-----**************')
            print()
            return
        else:
            print('OK')

        # 根据连通域大小排序
        # 同时滤除掉大于最大像素阈值面积、小于最小像素阈值阈值的区域
        area_idx = np.argsort(area)
        print("Area indexs are:", area_idx)
        print("After sorted, area pixel nums are:", area[area_idx])
        sorted_areas = area[area_idx]

        valid_idx = (sorted_areas >= min_area_pixels) & (sorted_areas <= max_area_pixels)
        sorted_areas = sorted_areas[valid_idx]

        print("Most possible {} areas are {}".format(nums, sorted_areas[-nums:]))
        contours = np.array(contours)[area_idx]
        print("Length of contours {}".format(len(contours)))
        contours = contours[valid_idx]
        print("Length of contours {}".format(len(contours)))
        final_contours = contours[-nums:]

        rect_points = []
        rect_center_points = []
        for con in final_contours:
            x, y, w, h = cv2.boundingRect(con)
            p1 = (x, y)
            p2 = (x + w, y + h)
            center_point = (x + w / 2, y + h / 2)
            rect_points.append((p1, p2))
            rect_center_points.append(center_point)

        # 返回 连通域边框, 框住的连通域最小矩形, 最小矩形中心点
        # list, list, list
        return final_contours, rect_points, rect_center_points


    def find_holes(self, nums=4, tube_volume=150, err_thres=5, tailored_size=(800, 800),
                         max_hole_area=None, demo_img_name='holes_with_multi_rectangles.jpg'):
        """
            找到一张图片中，可能的可以放入试管的槽位
            根据所需要的试管大小，给出对应孔洞像素坐标
            :param centri_img:  输入的离心机内部图片
            :param center_pos:  离心机中心坐标
            :param tube_volume: 所需要的试管容积
            :param nums:        需要返回的试管孔数目
            :param err_thres:   检查返回试管坐标时, 允许的非对称像素误差
            :return:    相应的坐标
            """
        # TODO: 由于离心机需要严格对称操作，所以最好一次性给出两个关于中心对称的坐标
        #  尚未设计出误差检测算法
        # 当前可用的图片编号为00024-00028

        center_pos = self.crop_center
        tailored_size = self.crop_size
        tailored_img = self.crop_img(crop_center=center_pos, crop_size=tailored_size)

        # 高斯平滑
        tailored_img = cv2.GaussianBlur(src=tailored_img, ksize=(5, 5), sigmaX=1)
        # 中值滤波
        tailored_img = cv2.medianBlur(src=tailored_img, ksize=5)
        thres_img = self.thres_binary_for_holes(img=tailored_img)

        # 存储二值化图像
        cv2.imwrite(os.path.join(self.temp_img_dir, 'holes_thres_img_0.jpg'), thres_img)
        # 找到最大联通区域，并且绘制轮廓
        contours, rect_points, rect_center_points = self.find_possible_areas(bi_img=thres_img, nums=nums,
                                                                             min_area_pixels=self.hole_min_pixels,
                                                                             max_area_pixels=self.holes_max_pixels)
        self.draw_rectangles(rect_points_list=rect_points, img_name=demo_img_name)
        # draw_multi_rectangles(src_img=tailored_img, rect_points_list=rect_points)

        print("Nums of detected contours: {}".format(len(contours)))
        print(rect_points)
        print("Bounding Rectangles' center points are")
        print(rect_center_points)

        real_points = []
        for item in rect_center_points:
            real_points.append(
                recover_pixel_coords(points=item, center_pos=self.crop_center, tailored_size=self.crop_size))
        print(real_points)

        return thres_img, rect_points, rect_center_points, real_points


    def find_tubes(self, nums=4, tube_volume=150, err_thres=5,
                   demo_img_name='tubdes_with_multi_rectangles.jpg', **kwargs):

        # TODO: 由于离心机需要严格对称操作，所以最好一次性给出两个关于中心对称的坐标
        # 当前可用的图片编号为00024-00028

        center_pos = self.crop_center
        tailored_size = self.crop_size
        tailored_img = self.crop_img(crop_center=center_pos, crop_size=tailored_size)

        thres_img = self.thres_binary_for_tubes(img=tailored_img, )
        # 为了减少连通区域的个数, 所做的必要的中值滤波
        thres_img = cv2.medianBlur(src=thres_img, ksize=5)
        # 存储二值化图像
        cv2.imwrite(os.path.join(self.temp_img_dir, 'tubes_thres_img_0.jpg'), thres_img)

        contours, rect_points, rect_center_points = self.find_possible_areas(bi_img=thres_img, nums=nums,
                                                                             min_area_pixels=self.tubes_max_pixels,
                                                                             max_area_pixels=self.tubes_max_pixels)
        self.draw_rectangles(rect_points_list=rect_points, img_name=demo_img_name)

        print("Nums of detected contours: {}".format(len(contours)))
        print(rect_points)
        print("Bounding Rectangles' center points are")
        print(rect_center_points)

        real_points = []
        for item in rect_center_points:
            real_points.append(
                recover_pixel_coords(points=item, center_pos=self.crop_center, tailored_size=self.crop_size))
        print(real_points)

        return thres_img, rect_points, rect_center_points, real_points






def centrifuge_capture(device_num=0, temp_dir='../temps'):
    """
    拍摄图片
    :param device_num:  摄像头设备号
    :param temp_dir:    照片临时存储文件夹
    :return:            拍摄的照片
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame


def thres_img_for_holes(img, bgr_thres=None):
    """
    for better to extract the rotor area, the suggested thresholds are:
        blue:   50 - 150
        green:  35 - 190
        red:    0 - 50
    :param img:
    :param bgr_thres:
    :return:
    """
    if bgr_thres is None:
        bgr_thres = [(0, 10), (0, 16), (0, 10)]
        # bgr_thres = [(0, 10), (0, 50), (0, 20)]
    b_thres, g_thres, r_thres = bgr_thres
    b_low, b_high = b_thres
    g_low, g_high = g_thres
    r_low, r_high = r_thres

    b, g, r = cv2.split(img)

    a_r = (r >= r_low) & (r <= r_high)
    a_g = (g >= g_low) & (g <= g_high)
    a_b = (b >= b_low) & (b <= b_high)

    thres = (a_r * a_g * a_b) * 255
    # cv2.imwrite('./temp.jpg', thres)
    # thres = cv2.imread('./temp.jpg', 0)

    return thres.astype(np.uint8)


def transfer_to_gray(img):
    # cv2.convert
    res = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    return res


def thres_img_for_tubes(img, bgr_thres=[(0, 50), (40, 150), (85, 210)]):
    """
    for better to localize the Orange Tubes, thresholds are suggested to be set at:
        blue:   0 - 180
        green:  150 - 255
        red:    200 - 255
    :param img:
    :param bgr_thres:
    :return:
    """
    b_thres, g_thres, r_thres = bgr_thres
    b_low, b_high = b_thres
    g_low, g_high = g_thres
    r_low, r_high = r_thres

    b, g, r = cv2.split(img)

    a_r = (r >= r_low) & (r <= r_high)
    a_g = (g >= g_low) & (g <= g_high)
    a_b = (b >= b_low) & (b <= b_high)

    thres = (a_r * a_g * a_b) * 255
    # cv2.imwrite('./temp.jpg', thres)
    # thres = cv2.imread('./temp.jpg', 0)

    return thres.astype(np.uint8)


def tailor_img(img, center_pos, size=(1100, 1100)):
    """
    根据所以的图片像素中心点，和 所需要的大小框，裁剪图片
    :param img:             可slice的对象
    :param center_pos:
    :param size:
    :return:
    """
    w, h, c = img.shape
    _h, _w = size

    assert w > _w
    assert h > _h

    half_h = _h // 2
    half_w = _w // 2
    center_y, center_x = center_pos

    upper_x = center_x - half_w
    upper_y = center_y - half_h
    if upper_x < 0 or upper_y < 0:
        print("Center Position is not Valid, please check it!")
        raise ValueError

    res = img[upper_x: upper_x + _w,
          upper_y: upper_y + _h]
    cv2.imwrite('../temps/tailored_img.jpg', res)
    return res

#
# def find_holes_canny(centri_img, center_pos, ):
#     t_img = tailor_img(img=centri_img, center_pos=center_pos)
#     canny = cv2.Canny(image=t_img, threshold1=40, threshold2=120)
#     cv2.imwrite('../temps/holes_canny_test_0.jpg', canny)
#     return canny


def find_possible_holes(centri_img, center_pos, nums=4,
                        tube_volume=150, err_thres=5, tailored_size=(800, 800),
                        max_hole_area = None):
    """
    找到一张图片中，可能的可以放入试管的槽位
    根据所需要的试管大小，给出对应孔洞像素坐标
    :param centri_img:  输入的离心机内部图片
    :param center_pos:  离心机中心坐标
    :param tube_volume: 所需要的试管容积
    :param err_thres:   检查返回试管坐标时, 允许的非对称像素误差
    :return:    相应的坐标
    """
    # TODO: 由于离心机需要严格对称操作，所以最好一次性给出两个关于中心对称的坐标
    # 当前可用的图片编号为00024-00028

    tailored_img = tailor_img(img=centri_img, center_pos=center_pos, size=tailored_size)
    # 高斯平滑
    tailored_img = cv2.GaussianBlur(src=tailored_img, ksize=(5, 5), sigmaX=1)
    # 中值滤波
    tailored_img = cv2.medianBlur(src=tailored_img, ksize=5)
    thres_img = thres_img_for_holes(img=tailored_img, bgr_thres=[(0, 10), (0, 13), (0, 10)])
    cv2.imwrite('../temps/holes_thres_img_0.jpg', thres_img)

    # 找到最大联通区域，并且绘制轮廓
    # contours, max_idx, (p1, p2) = find_max_contours(bi_img=thres_img, src_img=tailored_img,
    #                                                 _save_path='../temps/__with_rect.jpg')

    contours, rect_points, rect_center_points = find_possible_areas(bi_img=thres_img, nums=nums,
                                                                    min_area_pixels=200, max_area_pixels=600)
    draw_multi_rectangles(src_img=tailored_img, rect_points_list=rect_points)

    print("Nums of detected contours: {}".format(len(contours)))
    print(rect_points)
    print("Bounding Rectangles' center points are")
    print(rect_center_points)

    # return thres_img
    return contours, rect_points, rect_center_points


def find_possible_tubes(centri_img, center_pos, nums=4,
                        tube_volume=150, err_thres=5, **kwargs):
    # TODO: 由于离心机需要严格对称操作，所以最好一次性给出两个关于中心对称的坐标
    # 当前可用的图片编号为00024-00028

    if 'tailored_size' not in kwargs.keys():
        tailored_size = (1100, 1100)
    else:
        tailored_size = kwargs['tailored_size']

    tailored_img = tailor_img(img=centri_img, center_pos=center_pos, size=tailored_size)
    thres_img = thres_img_for_tubes(img=tailored_img, )
    # 为了减少连通区域的个数, 所做的必要的中值滤波
    thres_img = cv2.medianBlur(src=thres_img, ksize=5)
    cv2.imwrite('../temps/tubes_thres_img_0.jpg', thres_img)

    # in find_max_contours, OpenCV will draw max rectangle area
    # contours, max_idx, (p1, p2) = find_max_contours(bi_img=thres_img, src_img=tailored_img,
    #                                                 _save_path='../temps/__with_rect.jpg')

    contours, rect_points, rect_center_points = find_possible_areas(bi_img=thres_img, nums=nums, min_area_pixels=100)
    draw_multi_rectangles(src_img=tailored_img, rect_points_list=rect_points)

    print("Nums of detected contours: {}".format(len(contours)))
    print(rect_points)
    print("Bounding Rectangles' center points are")
    print(rect_center_points)

    return thres_img, rect_points, rect_center_points


def find_possible_areas(bi_img, nums=4, min_area_pixels=200, max_area_pixels=1150):
    _bi_img = bi_img[:, :]
    print(_bi_img.shape)

    contours, hierarchy = cv2.findContours(_bi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours num is {}".format(len(contours)))
    area = []

    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    area = np.array(area)

    if len(area) == 0:
        print('***************-----**************')
        print()
        return
    else:
        print('OK')
    # max_idx = np.argmax(area)
    # print("max area num is {}".format(area[max_idx]))

    # filter area, using area pixel num
    # maintain the areas whose pixels sum is between [min, max]


    area_idx = np.argsort(area)
    print("Area indexs are:", area_idx)
    print("After sorted, area pixel nums are:", area[area_idx])
    sorted_areas = area[area_idx]

    valid_idx = (sorted_areas >= min_area_pixels) & (sorted_areas <= max_area_pixels)
    sorted_areas = sorted_areas[valid_idx]

    print("Most possible {} areas are {}".format(nums, sorted_areas[-nums:]))
    contours = np.array(contours)[area_idx]
    print("Length of contours {}".format(len(contours)))
    contours = contours[valid_idx]
    print("Length of contours {}".format(len(contours)))
    final_contours = contours[-nums:]

    rect_points = []
    rect_center_points = []
    for con in final_contours:
        # minRect = cv2.minAreaRect(con)
        x, y, w, h = cv2.boundingRect(con)
        p1 = (x, y)
        p2 = (x + w, y + h)
        center_point = (x + w / 2, y + h / 2)
        rect_points.append((p1, p2))
        rect_center_points.append(center_point)

    return final_contours, rect_points, rect_center_points

    # 用来寻找二值化图像中的可能区域


def synthetic_detection(area_pos_list, centri_pos=(800, 800), err_range=5):
    # 根据返回的中心点list，找到对称的两点, 并根据离心机中心坐标进行纠正
    pass


def draw_multi_rectangles(src_img, rect_points_list, save_path='../temps/__with_multi_rectangles.jpg'):
    length = len(rect_points_list)

    # draw rectangles
    for i in range(length):
        (pt1, pt2) = rect_points_list[i]

        cv2.rectangle(img=src_img, pt1=pt1, pt2=pt2, color=(183, 152, 63), thickness=2)
    s = cv2.imwrite(save_path, src_img)
    if not s:
        print('Not saved')
        print("Please check if the path is right or if the directory exists")
        raise ValueError

# def undistortion(img,):
#     carlibrator = CameraCalibrator()

def load_camera_params(param_file: str = 'camera_params.yaml'):
    if not os.path.exists(param_file):
        print("File {} does not exist.", format(param_file))
        exit(-1)
    with open(param_file, 'r') as f:
        param = yaml.load(f)
    matrix = np.array(param['camera_matrix'])
    new_camera_matrix = np.array(param['new_camera_matrix'])
    dist = np.array(param['camera_distortion'])
    image_size = (param['image height'], param['image width'])
    roi = np.array(param['roi'])

    return [matrix, new_camera_matrix, dist, image_size, roi]

def undistortion(img, camera_params):
    """

    :param img:
    :param camera_params:
    :return:
    """
    [matrix, new_camera_matrix, dist, image_size, roi] = camera_params
    if not isinstance(img, np.ndarray):
        AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
    img_w, img_h, img_c = img.shape

    dst = cv2.undistort(img, matrix, dist, None, new_camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    dst = cv2.resize(dst, (img_h, img_w), interpolation=cv2.INTER_CUBIC)
    return dst

def recover_pixel_coords(points, center_pos=(390, 240), tailored_size=(425, 425)):
    w, h = tailored_size
    t_cen = (w/2, h/2)
    x = points[0] - t_cen[0] + center_pos[0]
    y = points[1] - t_cen[1] + center_pos[1]
    return (x, y)


def find_holes_using_depth(depth_file_path, center_pos=None, crop_size=None, ROI=None,
                           temp_depth_img_path='../temps/centri_depth_norm.jpg'):
    """

    :param depth_file_path: str, a .npy file
    :param center_pos:
    :param crop_size:
    :return:
    """
    depth = np.load(depth_file_path)
    w, h = depth.shape
    print("depth image shape:\t", depth.shape)

    upper_x, upper_y = ROI[0]
    lower_x, lower_y = ROI[1]
    depth = depth[upper_x:lower_x, upper_y:lower_y]

    MIN = np.mean(depth)
    MAX = np.max(depth)

    _depth = ((depth - MIN) / (MAX - MIN)) * 255
    _depth = np.array(_depth, dtype=np.uint8)
    cv2.imwrite(temp_depth_img_path, _depth)

    return depth







def main():
    color_img_dir = '../centri_doc/color/'

    tubes_img = cv2.imread('../datas/centrifuges/color/color_1603163102.1412394.jpg')
    tubes_img = cv2.imread('../datas/centrifuges/color/color_1603163004.5233963.jpg')
    tubes_img = cv2.imread('../datas/centrifuges/color/color_1604648856.jpg')
    # empty_img = cv2.imread('../datas/centrifuges/color/color_1604566870.jpg')
    empty_img = cv2.imread('../datas/centrifuges/color/empty-3.jpg')
    # empty_img = cv2.imread('../datas/centrifuges/color/color_1603162977.0726545.jpg')

    simple_tube_img = cv2.imread('../')

    # mm = find_holes_canny(centri_img=empty_img, center_pos=(824, 768))



    # tubes position
    thres_tubes, rect_points, rect_center_points = find_possible_tubes(centri_img=tubes_img, center_pos=(203, 222),
                                                                       tailored_size=(406, 406))
    r_points = []
    for item in rect_center_points:
        r_points.append(recover_pixel_coords(points=item, center_pos=(203, 222), tailored_size=(406, 406)))
    print(r_points)
    # camera_params = load_camera_params()
    # test_dst = undistortion(img=tubes_img, camera_params=camera_params)

    # cv2.imwrite('../temps/test_undst_img_1.jpg', test_dst)
    print('\n***************\n')

    # holes position
    thres_holes, rect_points, rect_center_points = find_possible_holes(centri_img=empty_img, center_pos=(203, 222),
                                                                       tailored_size=(406, 406), nums=4)
    r_points = []
    for item in rect_center_points:
        r_points.append(recover_pixel_coords(points=item, center_pos=(203, 222), tailored_size=(406, 406)))
    print(r_points)

    # 深度摄像头与彩色摄像头之间有平移
    holes_depth = find_holes_using_depth(depth_file_path='../datas/centrifuges/color/empty-1.npy',
                                         center_pos=(247, 210), crop_size=(366, 346), ROI=[[27, 64], [383, 430]])

    thermos_for_tubes = Centrifuge(color_img_path='../datas/centrifuges/color/color_1604648856.jpg',
                                   depth_npy_path='../datas/centrifuges/color/empty-1.npy')
    thermos_for_holes = Centrifuge(color_img_path='../datas/centrifuges/color/empty-3.jpg',
                                   depth_npy_path='../datas/centrifuges/color/empty-1.npy')

    thres_holes, rect_points, rect_center_points, real_points = thermos_for_holes.find_holes(nums=4, )
    # thres_holes, rect_points, rect_center_points, real_points = thermos_for_tubes.find_tubes(nums=4, )







#TODO:
# 1. When find the holes, try to set a reasonable area pixels, to filter the 'unvalid' holes
#    which are too large or small
# 2. Using depth information to detect holes






if __name__ == '__main__':
    main()
