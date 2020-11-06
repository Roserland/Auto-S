# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
import yaml
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import sys
from copy import deepcopy

class CameraCalibrator(object):
    def __init__(self, image_size: tuple, load=False, param_file='camera_params.yaml', \
                 image_data=None, image_points=None, depth=None, world_points=None, \
                 corner_width=None, corner_height=None, square_size=None):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        # camera intrinsic and distortion param.
        self.matrix = np.zeros((3, 3), np.float)
        self.new_camera_matrix = np.zeros((3, 3), np.float)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int)
        # world coordinate system to camera coordinate system.
        self.rotation_t_Homogeneous_matrix = np.zeros((4, 4))
        self.rvec = np.zeros((3, 1))
        self.tvec = np.zeros((3, 1))
        self.scale = 0
        self.cam2base_affine = np.zeros((3, 4))
        if load and os.path.exists(param_file):
            self.load_params(param_file=param_file)
        else:
            self.calibration(image_data, image_points, depth, world_points, corner_width, corner_height, square_size)
            self.save_params(param_file)

    def load_params(self, param_file: str = 'camera_params.yaml'):
        if not os.path.exists(param_file):
            print("File {} does not exist.", format(param_file))
            exit(-1)
        with open(param_file, 'r') as f:
            param = yaml.load(f)

        self.image_size = (param['image height'], param['image width'])
        # camera intrinsic and distortion param
        self.matrix = np.array(param['camera_matrix'])
        self.new_camera_matrix = np.array(param['camera new_camera_matrix'])
        self.dist = np.array(param['camera_distortion'])
        self.roi = np.array(param['camera roi'])
        # world coordinate system to camera coordinate system.
        self.rotation_t_Homogeneous_matrix = np.array(param['camera rotation_t_Homogeneous_matrix'])
        self.rvec = np.array(param['camera rvec'])
        self.tvec = np.array(param['camera tvec'])
        self.scale = param['camera scale']
        # camera coordinate system to robot base coordinate system with rigid affine.
        self.cam2base_affine = np.array(param['camera affine'])

    def save_params(self, save_path='camera_params.yaml'):
        height = self.image_size[0]
        width = self.image_size[1]
        # camera intrinsic and distortion param
        mat = [[float(self.matrix[row][column]) \
                for column in range(self.matrix.shape[1])] for row in range(self.matrix.shape[0])]
        new_mat = [[float(self.new_camera_matrix[row][column]) \
                    for column in range(self.new_camera_matrix.shape[1])] for row in
                   range(self.new_camera_matrix.shape[0])]
        dist = [[float(self.dist[row][column]) \
                 for column in range(self.dist.shape[1])] for row in range(self.dist.shape[0])]
        roi = [int(self.roi[i]) for i in range(self.roi.shape[0])]
        # world coordinate system to camera coordinate system.
        rotation_t = [[float(self.rotation_t_Homogeneous_matrix[row][column]) \
                       for column in range(self.rotation_t_Homogeneous_matrix.shape[1])] for row in
                      range(self.rotation_t_Homogeneous_matrix.shape[0])]
        rvec = [[float(self.rvec[row][column]) \
                 for column in range(self.rvec.shape[1])] for row in range(self.rvec.shape[0])]
        tvec = [[float(self.tvec[row][column]) \
                 for column in range(self.tvec.shape[1])] for row in range(self.tvec.shape[0])]
        scale = float(self.scale)
        # camera coordinate system to robot base coordinate system with rigid affine.
        cam2base_affine = [[float(self.cam2base_affine[row][column]) \
                            for column in range(self.cam2base_affine.shape[1])] for row in
                           range(self.cam2base_affine.shape[0])]

        records = {
            'image height': height,
            'image width': width,
            'camera_matrix': mat,
            'camera new_camera_matrix': new_mat,
            'camera_distortion': dist,
            'camera roi': roi,
            'camera rotation_t_Homogeneous_matrix': rotation_t,
            'camera rvec': rvec,
            'camera tvec': tvec,
            'camera scale': scale,
            'camera affine': cam2base_affine,
        }
        with open(save_path, 'w') as f:
            yaml.dump(records, f)
        print("Saved params in {}.".format(save_path))

    def _draw(self, img, obj_corners, img_corners=None, save_path='/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/coordinate.jpg'):
        print('IN DRAW- FUNCTIONS\n')
        # corners = np.vstack([obj_corners[0], obj_corners[28], obj_corners[36], obj_corners[53]])
        corners = np.vstack([obj_corners[0], [3, 0, 0], [0, 3, 0], [0, 0, -3]])
        corner_center = obj_corners[0]
        corner_z = np.array([corner_center[0], corner_center[1], corner_center[2] + 96])
        corner_x = corners[2]   #fzw
        corner_y = corners[1]   #fzw
        # corner_x = np.array([corner_center[0] + 48, corner_center[1], corner_center[2]])
        # corner_y = np.array([corner_center[0], corner_center[1] + 48, corner_center[2]])
        corners = np.vstack([corner_center, corner_x, corner_y, corner_z])
        print("\n in Draw Functions, the R-Matrix and T-Matrix are:\n")
        print(self.rotation_t_Homogeneous_matrix)
        print(self.tvec, '\n')
        imgpts, jac = cv2.projectPoints(corners, self.rvec, self.tvec, self.matrix, self.dist)

        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()), tuple(imgpts[1].astype(np.int).ravel()),
                       (255, 0, 0), 2) # x, blue
        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()), tuple(imgpts[2].astype(np.int).ravel()),
                       (0, 255, 0), 2) # y, green
        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()),
                       tuple(imgpts[3].astype(np.int).astype(np.int).ravel()), (0, 0, 255), 2)
        cv2.imwrite(save_path, img)    # z, red
        print("drawing point")
        if img_corners is not None:
            for point in img_corners:
                print(point)
                # _point = point[0]
                cv2.circle(img, center=tuple(point[0].astype(np.int).ravel()),
                           radius=2, color=(90, 100, 200), thickness=4)
        cv2.imwrite(save_path, img)    # z, red
        return img

    def calibration(self, file_path=None, image_points=None, depth=None, world_points=None, corner_width: int = 9,
                    corner_height: int = 6, square_size: float = 24.0):
        obj_corners = self._cal_real_corner(corner_width, corner_height, square_size)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret = self._calibrate_camera_intrinsic(file_path, obj_corners, corner_width, corner_height, square_size,
                                               criteria)
        self._calibrate_camera_extrinsic(image_points, depth, world_points, corner_width, corner_height, square_size,
                                         criteria)
        return ret

    # corner_width = 9, corner_height = 6;
    def _cal_real_corner(self, corner_width, corner_height, square_size):
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)  # (w*h)*2
        return obj_corner * square_size


    def _world2image(self, world_point=None):
        # 3D to 2D, real_point = np.array([[Xw], [Yw], [Zw]], dtype=np.float)
        imgpts, jac = cv2.projectPoints(world_point.transpose(1, 0), self.rvec, self.tvec, self.matrix, self.dist)
        return imgpts

    def _image2world_pnp(self, image_point=None, zConst_W=None, zConst_C=None):
        """
        image_point: np.array([[u], [v]], dtype=np.float)
        zConst_W: depth in world coordinate system.
        solve the scale factor s / zConst_C.
        """
        assert (not zConst_W is None) or (not zConst_C is None)
        # 3 x 3 rotation matrix
        revc_M1 = self.rotation_t_Homogeneous_matrix[0:3, 0:3]
        if zConst_C is None:
            image_point = np.array([image_point[0].tolist() + [1]])
            tempMat1 = np.linalg.inv(revc_M1).dot(np.linalg.inv(self.matrix)).dot(image_point.T)
            tempMat1 = np.matrix(tempMat1)
            tempMat2 = np.linalg.inv(revc_M1).dot(self.tvec)
            tempMat2 = np.matrix(tempMat2)
            print(tempMat2, tempMat1)
            s = (zConst_W + tempMat2[2, 0]) / tempMat1[2, 0]
            print(tempMat2[2, 0], tempMat1[2, 0])
            print("Calclulated Scale Factor : {}\n".format(s))
        else:
            s = zConst_C
        x = image_point[0, 0]
        y = image_point[0, 1]
        image_point = np.array([[x], [y], [1]])
        wc_point = np.dot(np.linalg.inv(revc_M1), (np.linalg.inv(self.matrix).dot(image_point) * s - self.tvec))
        return wc_point


    def img2world(self, img_point, z_constS):
        """
        without distortion, just a test
        :param img_point:
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

        rotation_mtx = self.rotation_t_Homogeneous_matrix[:3, :3]
        translation_mtx = self.tvec
        # rotation_mtx = self.rotation_mtx
        # translation_mtx = self.translation_mtx
        camera_mtx = self.matrix
        rotation_mtx_I = np.linalg.inv(rotation_mtx)
        camera_mtx_I = np.linalg.inv(camera_mtx)
        translation_mtx = np.array(translation_mtx)

        temp_mtx = np.dot(camera_mtx_I, _img_point.T) * z_constS - translation_mtx

        world_3d = rotation_mtx_I.dot(temp_mtx)

        return world_3d


imgs_dir_for_calcu_intrinsic = '/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/datas/carlibration/color/'
imgs_dir_for_calcu_extrinsic = '/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/datas/carlibration/'
c_h = 9 # corner height
c_w = 6 # corner width
square = 24
img_for_extrinsic = '/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/v-chessboard-5.jpg'
temp_save_dir = '/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/temps/chessboards/for_calibrations/'



def carlibration(imgs_dir=imgs_dir_for_calcu_intrinsic, world_points=None, c_w=6, c_h=9, square_size=24,
                 if_show=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((c_w * c_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:c_h, 0:c_w].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(os.path.join(imgs_dir_for_calcu_intrinsic, '*.jpg'))

    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (c_h, c_w), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            if if_show:
                img = cv2.drawChessboardCorners(img, (c_h, c_w), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print(mtx)
    # print(dist)
    # print(rvecs)
    # print(tvecs)

    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    print("total error: ", tot_error / len(objpoints))
    print("calibration over")
    print('##########################################\n')

    return mtx, dist

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 2)
    return img

def _extrinsic_singe(c_w=6, c_h=9):
    for fname in glob.glob('left*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (c_h, c_w), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            # cv2.drawChessboardCorners(img, (c_h, c_w), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

            img = draw(img, corners2, imgpts)
            cv2.imshow('img', img)
            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(fname[:6] + '.png', img)

    cv2.destroyAllWindows()

def calcu_abs_coords(point_arr, x_vec, y_vec):
    """
    calculate coordinates at x,y axis
    :param point:
    :param x_vec:
    :param y_vec:
    :return:
    """
    A = np.array([x_vec, y_vec])
    A = np.matrix(A).T
    b = np.matrix(point_arr).T

    # print("A shape:", A.shape)
    # print("b shape:", b.shape)

    res = np.linalg.solve(A, b)
    return res

def coordinate_transformer_2D(aim_img_point, image_basic_points, world_basic_points,):
    """
    transfer a image-2D point to the world-2D point
    :param aim_img_point:           the image point will be transformed into world object point：1x2 np.array
    :param image_basic_points:      image coordinates, which include origin, x_axis unit vector, y_axis unit vector
    :param world_basic_points:      world coordinates, which include origin, x_axis unit vector, y_axis unit vector
    :return:
    """
    img_o, img_x1, img_y1 = np.array(image_basic_points)
    w_o, w_x1, w_y1 = np.array(world_basic_points)
    # print('in coordinate transformer:')
    # print(img_o, img_x1, img_y1)
    # print(w_o, w_x1, w_y1)

    img_x_vec = img_x1 - img_o
    img_y_vec = img_y1 - img_o
    world_x_vec = w_x1 - w_o
    world_y_vec = w_y1 - w_o
    # print("basic vecs:")
    # print(img_x_vec, '\t', img_y_vec)

    aim_img_point = np.array(aim_img_point)
    abs_x_img_vec = aim_img_point - img_o
    # print("aim vecs:")
    # print(abs_x_img_vec)

    abs_coords = calcu_abs_coords(point_arr=abs_x_img_vec, x_vec=img_x_vec, y_vec=img_y_vec)
    # print("coordinates:\t", abs_coords)

    res = abs_coords[0] * world_x_vec + abs_coords[1] * world_y_vec
    # print("res: ", res)
    res = res[0][:] + w_o
    # print("res: ", res)
    return res

def coordinate_trans_using4origin(aim_img_point, image_basic_coords_4, world_basic_points_4,):
    assert len(image_basic_coords_4) == 4
    assert len(world_basic_points_4) == 4

    # [image_basic_points_0, image_basic_points_1,
    #  image_basic_points_2, image_basic_points_3, ] = image_basic_coords_4
    #
    # [world_basic_points_0, world_basic_points_0,
    #  world_basic_points_0, world_basic_points_0, ] = world_basic_points_4

    res = []
    for i in range(4):
        temp_image_basic_points = image_basic_coords_4[i]
        temp_world_basic_points = world_basic_points_4[i]

        temp_res = coordinate_transformer_2D(aim_img_point=aim_img_point, image_basic_points=temp_image_basic_points,
                                             world_basic_points=temp_world_basic_points)
        res.append(temp_res.tolist())
    res = np.array(res)

    return res.mean(axis=0)

def add_offset():
    pass

def construct_chessboard_coord(aim_img_point, obj_corners, chess_img_corners,
                               measured_points, using_coords_num=4):
    assert obj_corners.shape[0]  == 54
    # using upper left corner as origin
    if not using_coords_num == 4 or using_coords_num == 2 or using_coords_num == 1:
        print("using_coords_num only can be set in [1, 2, 4")

    obj_coord_upper_left = [obj_corners[0], obj_corners[1], obj_corners[9]]
    img_coord_upper_left = [chess_img_corners[0], chess_img_corners[1], chess_img_corners[9]]
    measured_upper_left  = [measured_points[0], measured_points[1], measured_points[9], ]
    # print("obj upper left:", obj_coord_upper_left)
    # print("measured upper left:", measured_upper_left)


    obj_coord_upper_right = [obj_corners[8], obj_corners[7], obj_corners[17]]
    img_coord_upper_right = [chess_img_corners[8], chess_img_corners[7], chess_img_corners[17]]
    measured_upper_right  = [measured_points[8], measured_points[7], measured_points[17], ]

    obj_coord_lower_left = [obj_corners[45], obj_corners[46], obj_corners[36]]
    img_coord_lower_left = [chess_img_corners[45], chess_img_corners[46], chess_img_corners[36]]
    measured_lower_left  = [measured_points[45], measured_points[46], measured_points[36], ]

    obj_coord_lower_right = [obj_corners[53], obj_corners[52], obj_corners[44]]
    img_coord_lower_right = [chess_img_corners[53], chess_img_corners[52], chess_img_corners[44]]
    measured_lower_right  = [measured_points[53], measured_points[52], measured_points[44], ]


    obj_coords = [obj_coord_upper_left, obj_coord_lower_right,
                  obj_coord_upper_right, obj_coord_lower_left]
    img_coords = [img_coord_upper_left, img_coord_lower_right,
                  img_coord_upper_right, img_coord_lower_left]
    measured_coords = [measured_upper_left, measured_upper_right,
                       measured_lower_left, measured_lower_right]

    res = []
    for i in range(using_coords_num):
        temp_world_point = coordinate_transformer_2D(aim_img_point=aim_img_point,
                                                     image_basic_points=img_coords[i],
                                                     world_basic_points=obj_coords[i])
        res.append(temp_world_point.tolist())
    res = np.array(res)
    res = res.mean(axis=0).ravel()[:2]
    # print("convert to obj_corners, the point is:\n", res)

    # finally, we should re-construct the point into real measured coordinates
    # create a 2D obj points
    _obj_corners = obj_corners[:, :2].reshape(chess_img_corners.shape)
    obj_coord_upper_left  = [_obj_corners[0],  _obj_corners[1], _obj_corners[9]]
    obj_coord_upper_right = [_obj_corners[8],  _obj_corners[7], _obj_corners[17]]
    obj_coord_lower_left  = [_obj_corners[45], _obj_corners[46], _obj_corners[36]]
    obj_coord_lower_right = [_obj_corners[53], _obj_corners[52], _obj_corners[44]]
    obj_coords_2d = [obj_coord_upper_left, obj_coord_upper_right,
                     obj_coord_lower_left, obj_coord_lower_right]

    m_res = []
    for i in range(4):
        _temps = coordinate_transformer_2D(aim_img_point=[res],
                                           image_basic_points=obj_coords_2d[i],
                                           # 此处用obj corner 代替 image corner的位置，会出现维度不一致的情况
                                           world_basic_points=measured_coords[i])
        m_res.append(_temps)
    m_res = np.array(m_res)
    # print("m_res", m_res)
    m_res = m_res.mean(axis=0).ravel()
    # print("converted measured points:\n", m_res)
    # print(m_res)

    return m_res


def img2world(img_point, camera_mtx,
              rotation_mtx,translation_mtx,
              z_constS):
    """
    without distortion, just a test
    :param img_point:
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
    rotation_mtx_I = np.linalg.inv(rotation_mtx)
    camera_mtx_I = np.linalg.inv(camera_mtx)
    translation_mtx = np.array(translation_mtx)

    temp_mtx = np.dot(camera_mtx_I, _img_point.T) * z_constS - translation_mtx

    world_3d = rotation_mtx_I.dot(temp_mtx)

    return world_3d

def coord_spanning(aim_point: np.array, origin: np.array, coefficient: float=0.849):
    """

    :param aim_point:
    :param origin:
    :param coefficient:
    :return:
    """
    vector = aim_point - origin
    res_pos = vector / coefficient + origin
    return res_pos



if __name__ == '__main__':
    mtx, dist = carlibration()

    mtx = np.array([[612.204, 0, 328.054],
                    [0, 611.238, 234.929],
                    [0,  0,  1]])
    # calibration over
    df = pd.read_csv('/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/标定new.csv',
                     header=None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((c_w * c_h, 3), np.float32)
    # objp = np.array(df.values[:, 12:15].astype(np.float) * 1000)
    # [[0.99710845, 0.07508602, -0.01169705]
    #  [-0.06578662,  0.92996514,  0.36171392]
    # [0.03803751, -0.3598985, 0.93221576]]
    measured_obj_corners_2D = np.array(df.values[:, 12:15].astype(np.float) * 1000)
    objp[:, :2] = np.mgrid[0:c_h, 0:c_w].T.reshape(-1, 2) * square
    objp = measured_obj_corners_2D.reshape(objp.shape)

    axis = np.float32([[1 * square, 0, 0], [0*square, 1*square, 0], [0, 0, -3 * square]]).reshape(-1, 3)

    # calculate extrinsic parameters
    depth = np.load("/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/v-chessboard-6.npy")
    for fname in glob.glob(os.path.join(imgs_dir_for_calcu_extrinsic, 'v*6.jpg')):
        print(fname)
        img = cv2.imread(fname)
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
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            r_mtx, _ = cv2.Rodrigues(rvecs)
            t_mtx = tvecs
            print("\nRotation matrix and Translation matrix:")
            print(r_mtx)
            print(t_mtx)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            imgp_x, imgp_y, imgp_z = imgpts
            print(imgp_x.ravel(), imgp_y.ravel(), imgp_z.ravel())
            img = draw(img, corners2, imgpts)
            print("O Points Pixels Pos:", corners2[13].ravel())
            # corners2[3]:345.1869   124.819336
            # corners2[13]:379.68304 138.98436
            cv2.imshow('img', img)
            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(os.path.join(temp_save_dir, fname[:6] + '.png'), img)
        else:
            print("Cannot get right return value")



    test_coods = coordinate_transformer_2D(aim_img_point=corners2[47].ravel(),
                              image_basic_points=[corners2[0], corners2[1], corners2[9]],
                              world_basic_points=[np.float32([0, 0, 0]), axis[0], axis[1]])
    print("tested coods:", test_coods)

    # re_coords = []
    # for i in range(54):
    #     point = corners2[i]
    #     test_coods = coordinate_transformer_2D(aim_img_point=point.ravel(),
    #                                            image_basic_points=[corners2[0], corners2[1], corners2[9]],
    #                                            # world_basic_points=[np.float32([0, 0, 0]), axis[0], axis[1]],
    #                                            world_basic_points=[objp[0], objp[1], objp[9]],
    #                                            )
    #     re_coords.append(test_coods)
    # re_coords = np.array(re_coords)
    # re_coords = re_coords.reshape(objp.shape)
    # print(np.round(re_coords - objp))
    # print('----------------------------------')
    # print('----------------------------------')
    #
    #
    # re_coords_2 = []
    # for i in range(54):
    #     point = corners2[i]
    #     test_coods = coordinate_transformer_2D(aim_img_point=point.ravel(),
    #                                            image_basic_points=[corners2[53], corners2[52], corners2[44]],
    #                                            world_basic_points=[objp[53], objp[52], objp[44]])
    #     re_coords_2.append(test_coods)
    # re_coords_2 = np.array(re_coords_2)
    # re_coords_2 = re_coords_2.reshape(objp.shape)
    # print(np.round(re_coords_2 - objp))
    #
    # res = (re_coords + re_coords_2) / 2
    # print('----------------------------------')
    # print('----------------------------------')
    # print(np.round(res - objp))
    #
    # re_coords_3 = []
    # for i in range(54):
    #     point = corners2[i]
    #     test_coods = coordinate_transformer_2D(aim_img_point=point.ravel(),
    #                                            image_basic_points=[corners2[45], corners2[46], corners2[36]],
    #                                            world_basic_points=[objp[45], objp[46], objp[36]])
    #     re_coords_3.append(test_coods)
    # re_coords_3 = np.array(re_coords_3).reshape(objp.shape)
    # print(np.round(re_coords_3 - objp))
    #
    # res = (re_coords + re_coords_2 + re_coords_3) / 3
    # print('----------------------------------')
    # print('----------------------------------')
    # print(np.round(res - objp))
    #
    # re_coords_4 = []
    # for i in range(54):
    #     point = corners2[i]
    #     test_coods = coordinate_transformer_2D(aim_img_point=point.ravel(),
    #                                            image_basic_points=[corners2[8], corners2[7], corners2[17]],
    #                                            world_basic_points=[objp[8], objp[7], objp[17]])
    #     re_coords_4.append(test_coods)
    # re_coords_4 = np.array(re_coords_4).reshape(objp.shape)
    #
    # res = (re_coords + re_coords_2 + re_coords_3 + re_coords_4) / 4
    # print('----------------------------------')
    # print('----------------------------------')
    # print(np.round(res - objp))


    df = pd.read_csv("/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/标定new.csv", header=None)
    measured_obj_corners_2D = np.array(df.values[:, 12:15].astype(np.float) * 1000)
    __pnp_points = []
    for i in range(len(corners2)):
        img_point = corners2[i]
        # temp_depth = depth[int(img_point[0, 1]), int(img_point[0, 0])]
        # real_height = measured_obj_corners_2D[i][2]
        # print(tvecs[2] - temp_depth, real_height)

        pnp_point = img2world(img_point=img_point[0], camera_mtx=mtx, rotation_mtx=r_mtx,
                              translation_mtx=t_mtx,
                              z_constS=depth[int(img_point[0, 1]), int(img_point[0, 0])])
        __pnp_points.append(pnp_point)
    __pnp_points = np.array(__pnp_points)

    # measured_obj_corners_2D[:, 2] = measured_obj_corners_2D[:, 2].mean()

    # for img_point in corners2:
    #     print(img_point)
    diffs = []
    for i in range(len(__pnp_points)):
        __mea = measured_obj_corners_2D[i].ravel()
        __cal = __pnp_points[i].ravel()
        diff = (__cal - __mea)
        print("diff [{}]:\t".format(i), np.round(diff)[:2])
        diffs.append(diff)
    # print(np.round(diffs))

    tubes_real_pos = np.array(pd.read_csv("/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/utils/carlibrations/tube_coord.csv", header=None)) * 1000
    tubes_real_pos = tubes_real_pos[:, 12:15]
    center = tubes_real_pos[0]
    tubes_real_pos = tubes_real_pos[1:5]
    tubes_height = tubes_real_pos[:, 2]
    tubes_mean_height = np.mean(tubes_height)
    print("tubes mean height: ", tubes_mean_height)

    # (362.5, 245.5), (291.0, 76.0), (193.5, 319.5), (122.0, 149.0)
    #       1              2            4                3
    tubes_detected_img_pos = [(362.5, 245.5), (291.0, 76.0), (122.0, 149.0), (193.5, 319.5)]

    # tubes_depth = np.load("/Users/fanzw/PycharmProjects/Others/Auto-S-Communications/datas/centrifuges/color/depth_1604562951.npy")
    center_img_pos = (236, 194)

    # calculate centrifuge center position
    cal_center_pos = img2world(img_point=center_img_pos, camera_mtx=mtx, rotation_mtx=r_mtx,
                              translation_mtx=t_mtx,
                              z_constS=tvecs[2] - tubes_mean_height)
    cal_center_pos = cal_center_pos.ravel()
    print("center diff:", cal_center_pos - center)

    tubes_detected_cal_pos = []
    for img_pos in tubes_detected_img_pos:
        temps = img2world(img_point=img_pos, camera_mtx=mtx, rotation_mtx=r_mtx,
                              translation_mtx=t_mtx,
                              z_constS=tvecs[2] - tubes_mean_height)
        tubes_detected_cal_pos.append(temps.ravel().tolist())
    tubes_detected_cal_pos = np.array(tubes_detected_cal_pos)
    print(tubes_detected_cal_pos - tubes_real_pos)

    tubes_detected_cal_pos[:, 2] = tubes_mean_height
    print(tubes_detected_cal_pos)


    # print(tubes_detected_cal_pos)

