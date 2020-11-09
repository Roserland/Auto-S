import numpy as np
import cv2
import os, yaml
import pandas as pd

class Centrifuge(object):
    def __init__(self, color_img_path, depth_npy_path,
                 crop_center=(203, 222), crop_size=(406, 406),
                 tubes_num=4, holes_num=4,
                 tubes_bgr_thres=None, holes_bgr_thres=None, temp_img_dir='../temps/centri_temp_imgs/',
                 tubes_min_pixels=100, tubes_max_pixels=1150, hole_min_pixels=200, holes_max_pixels=480):
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
        return (int(x), int(y))

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
        assert min_area_pixels < max_area_pixels

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


    def find_holes(self, nums=4, tube_volume=150, err_thres=5,
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
        # visualization: draw multiple rectangles.
        new_rect_points = []
        for item in rect_points:
            p1, p2 = item
            p1 = self.coords_transformer(points=p1, center_pos=self.crop_center, tailored_size=self.crop_size)
            p2 = self.coords_transformer(points=p2, center_pos=self.crop_center, tailored_size=self.crop_size)
            new_rect_points.append((p1, p2))
        self.draw_rectangles(rect_points_list=new_rect_points, img_name=demo_img_name)

        print("Nums of detected contours: {}".format(len(contours)))
        print(rect_points)
        print("Bounding Rectangles' center points are")
        print(rect_center_points)

        real_points = []
        for item in rect_center_points:
            real_points.append(
                self.coords_transformer(points=item, center_pos=self.crop_center, tailored_size=self.crop_size))
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
                                                                             min_area_pixels=self.tubes_min_pixels,
                                                                             max_area_pixels=self.tubes_max_pixels)

        print("Nums of detected contours: {}".format(len(contours)))
        print(rect_points)
        print("Bounding Rectangles' center points are")
        print(rect_center_points)

        real_points = []
        for item in rect_center_points:
            real_points.append(
                self.coords_transformer(points=item, center_pos=self.crop_center, tailored_size=self.crop_size))
        print(real_points)

        new_rect_points = []
        for item in rect_points:
            p1, p2 = item
            p1 = self.coords_transformer(points=p1, center_pos=self.crop_center, tailored_size=self.crop_size)
            p2 = self.coords_transformer(points=p2, center_pos=self.crop_center, tailored_size=self.crop_size)
            new_rect_points.append((p1, p2))
        self.draw_rectangles(rect_points_list=new_rect_points, img_name=demo_img_name)

        return thres_img, rect_points, rect_center_points, real_points


class Realsense_Calibrator(object):
    def __init__(self, chessboard_color_path=None, chessboard_depth_path=None,
                 calibration=False):
        # camera intrinsic matrix
        self.c_mtx = np.array([[612.204, 0, 328.054],
                                [0, 611.238, 234.929],
                                [0,  0,  1]])
        self.dist = np.array([[0, 0, 0, 0, 0]])
        self.r_mtx = np.matrix([[-0.26458059, 0.96286004, 0.05382989],
                                [0.95508371,  0.26935109, -0.12355197],
                                [-0.13346239, 0.0187226,  -0.99087701]])
        self.t_mtx = np.matrix([[462.7425878, ],
                                [40.05304244, ],
                                [879.0725397, ]])
        self.chessboard_color_path = chessboard_color_path
        self.chessboard_depth_path = chessboard_depth_path
        self.calibration = calibration

    def img2world(self, img_point, zConst_S):
        """
            without distortion, just a test
            :param img_point: 1 x 2 array
            :param zConst_s:
            :return:
            """
        if len(img_point) == 2:
            _img_point = np.array([[img_point[0], img_point[1], 1]])
        elif len(img_point) == 3:
            _img_point = np.array([img_point])
        else:
            print("please check the image coordinates")
            raise ValueError
        rotation_mtx_I = np.linalg.inv(self.r_mtx)
        camera_mtx_I = np.linalg.inv(self.c_mtx)
        translation_mtx = np.array(self.t_mtx)

        temp_mtx = np.dot(camera_mtx_I, _img_point.T) * zConst_S - translation_mtx

        world_3d = rotation_mtx_I.dot(temp_mtx)

        return world_3d

    def re_calibration(self, c_w=6, c_h=9, square=24,
                       real_pos_file='/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/标定new.csv',):
        """
        calibrating a single chessboard image whose corners have correlated real world position;
        mainly for extrinsic parameters;
        :return:
        """
        chessboard_color_img = cv2.imread(self.chessboard_color_path)
        chessboard_depth_img = cv2.imread(self.chessboard_depth_path)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((c_w * c_h, 3), np.float32)
        df = pd.read_csv(real_pos_file, header=None)
        measured_obj_corners_2D = np.array(df.values[:, 12:15].astype(np.float) * 1000)
        objp[:, :2] = np.mgrid[0:c_h, 0:c_w].T.reshape(-1, 2) * square
        objp = measured_obj_corners_2D.reshape(objp.shape)

        img = chessboard_color_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (c_h, c_w), None)
        print(ret)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print('basic corner point:\n', corners2[0], corners2[1], corners2[9])
            cv2.drawChessboardCorners(img, (c_h, c_w), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, self.c_mtx, self.dist)

            r_mtx, _ = cv2.Rodrigues(rvecs)
            t_mtx = tvecs
            print("\nRotation matrix and Translation matrix:")
            print(r_mtx)
            print(t_mtx)

        else:
            print("Cannot get right return value")

        self.r_mtx = r_mtx
        self.t_mtx = t_mtx

    def coor_transform(self, img_point, zConst_s):
        if self.calibration:
            self.re_calibration()

        res = self.img2world(img_point=img_point, zConst_S=zConst_s)
        return res.ravel()






def main():
    # color_img_dir = '../centri_doc/color/'
    #
    # tubes_img = cv2.imread('../datas/centrifuges/color/color_1603163102.1412394.jpg')
    # tubes_img = cv2.imread('../datas/centrifuges/color/color_1603163004.5233963.jpg')
    # tubes_img = cv2.imread('../datas/centrifuges/color/color_1604648856.jpg')
    # empty_img = cv2.imread('../datas/centrifuges/color/color_1604566870.jpg')
    # empty_img = cv2.imread('../datas/centrifuges/color/empty-3.jpg')
    # empty_img = cv2.imread('../datas/centrifuges/color/color_1603162977.0726545.jpg')

    # simple_tube_img = cv2.imread('../')

    thermos_for_tubes = Centrifuge(color_img_path='../datas/centrifuges/color/color_1604648856.jpg',
                                   depth_npy_path='../datas/centrifuges/color/empty-1.npy')
    thermos_for_holes = Centrifuge(color_img_path='../datas/centrifuges/color/empty-3.jpg',
                                   depth_npy_path='../datas/centrifuges/color/empty-1.npy')

    # thres_holes, rect_points, rect_center_points, real_points = thermos_for_holes.find_holes(nums=4, )
    print('\n------------------------------------------------\n')
    thres_holes, rect_points, rect_center_points, real_points = thermos_for_tubes.find_tubes(nums=4, )
    print(real_points)

    coordinates_transformer = Realsense_Calibrator()
    tubes_detected_img_pos = real_points
    tubes_mean_height = 236.8475970086949
    tvecs = coordinates_transformer.t_mtx
    tubes_detected_cal_pos = []
    for img_pos in tubes_detected_img_pos:
        temps = coordinates_transformer.coor_transform(img_point=img_pos, zConst_s=tvecs[2] - tubes_mean_height)
        tubes_detected_cal_pos.append(temps.ravel().tolist())
    tubes_detected_cal_pos = np.array(tubes_detected_cal_pos)
    print(tubes_detected_cal_pos)



if __name__ == '__main__':
    main()