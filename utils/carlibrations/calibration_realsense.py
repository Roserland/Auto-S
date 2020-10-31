# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
import yaml
import xml.etree.ElementTree as ET
import argparse
import pandas as pd


class CameraCalibrator(object):
    def __init__(self, image_size: tuple, chessboards_dir):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float)
        self.new_camera_matrix = np.zeros((3, 3), np.float)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int)
        self.chessborads_dir = chessboards_dir
        self.scale = 0

    def load_params(self, param_file: str = 'camera_params.yaml'):
        if not os.path.exists(param_file):
            print("File {} does not exist.", format(param_file))
            exit(-1)
        with open(param_file, 'r') as f:
            param = yaml.load(f)

        self.matrix = np.array(param['camera_matrix'])
        self.new_camera_matrix = np.array(param['new_camera_matrix'])
        self.dist = np.array(param['camera_distortion'])
        self.image_size = (param['image height'], param['image width'])
        self.roi = np.array(param['roi'])

    def save_params(self, save_path='camera_params.yaml'):
        mat = [[float(self.matrix[row][column]) \
                for column in range(self.matrix.shape[1])] for row in range(self.matrix.shape[0])]
        new_mat = [[float(self.new_camera_matrix[row][column]) \
                    for column in range(self.new_camera_matrix.shape[1])] for row in
                   range(self.new_camera_matrix.shape[0])]
        dist = [float(self.dist[0][i]) for i in range(self.dist.shape[1])]
        roi = [int(self.roi[i]) for i in range(self.roi.shape[0])]
        height = self.image_size[0]
        width = self.image_size[1]
        rvecs = self.rvecs
        tveces = self.tveces
        records = {
            'image height': height,
            'image width': width,
            'camera_matrix': mat,
            'new_camera_matrix': new_mat,
            'camera_distortion': dist,
            'roi': roi,
            'rvecs': rvecs,
            'tveces': tveces
        }
        with open(save_path, 'w') as f:
            yaml.dump(records, f)
        print("Saved params in {}.".format(save_path))

    # corner_width = 9, corner_height = 6;
    def _cal_real_corner(self, corner_width, corner_height, square_size):
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)  # (w*h)*2
        return obj_corner * square_size

    def calibration(self, corner_width: int, corner_height: int, square_size: float):
        file_names = glob.glob(os.path.join(self.chessborads_dir, '*.JPG')) + \
                     glob.glob(os.path.join(self.chessborads_dir, '*.jpg')) + \
                     glob.glob(os.path.join(self.chessborads_dir, '*.png'))
        objs_corner = []  # 3d point in real world space.
        imgs_corner = []  # 2d point in image plane.

        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数 30 和最大误差容限 0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 获取标定板角点的位置
        obj_corner = self._cal_real_corner(corner_width, corner_height, square_size)

        for file_name in file_names:
            # read image
            chess_img = cv2.imread(file_name)
            print(chess_img.shape)
            assert (chess_img.shape[0] == self.image_size[0] and chess_img.shape[0] == self.image_size[0]), \
                "Image size does not match the given value {}.".format(self.image_size)
            # to gray
            gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv2.findChessboardCorners(gray, (corner_width, corner_height))

            # If found, add object points, image points (after refining them).
            if ret:
                objs_corner.append(obj_corner)
                img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(square_size // 2, square_size // 2),
                                               zeroZone=(-1, -1), criteria=criteria)
                imgs_corner.append(img_corners)
                # import pdb; pdb.set_trace()

                # Draw and display the corners.
                cv2.drawChessboardCorners(gray, (corner_width, corner_height), img_corners, ret)
                if not os.path.exists('./chessboard'):
                    os.makedirs('./chessboard')
                cv2.imwrite(os.path.join('./chessboard', file_name.split('/')[-1]), gray)
            else:
                print("Fail to find corners in {}.".format(file_name))
        # calibration
        ret, self.matrix, self.dist, rvecs, tveces = cv2.calibrateCamera(objs_corner, imgs_corner, gray.shape[::-1],
                                                                         None, None)
        # matrix, 内参数矩阵 （3 * 3）
        # dist, 畸变系数（k1,k1,k3,p1,p2)
        # revecs 旋转向量， 外参数
        # tveces 平移向量， 外参数
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, gray.shape[::-1], alpha=0,
                                                                    newImgSize=gray.shape[::-1])
        self.roi = np.array(roi)

        self.rvecs = [arr.tolist() for arr in rvecs]
        self.tveces = [arr.tolist() for arr in tveces]
        # print('epoch', rvecs, "\n")
        # 计算重投影误差
        total_error = 0
        for index in range(len(objs_corner)):
            img_corner2, _ = cv2.projectPoints(objs_corner[index], rvecs[index], tveces[index],
                                               self.matrix, self.dist)
            error = cv2.norm(imgs_corner[index], img_corner2, cv2.NORM_L2) / len(img_corner2)
            total_error += error
            print(file_name, ': ', error)
        print("mean projection error: ", total_error / len(objs_corner))
        return ret

    def carlibrate_single(self, img_path, corner_width: int, corner_height: int, square_size: float):
        """
        标定单张棋盘格图片，主要为了得到其旋转矩阵和平移矩阵
        :return:
        """
        chess_img = cv2.imread(img_path)
        objs_corner = []  # 3d point in real world space.
        imgs_corner = []  # 2d point in image plane.

        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数 30 和最大误差容限 0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 获取标定板角点的位置
        obj_corner = self._cal_real_corner(corner_width, corner_height, square_size)

        assert (chess_img.shape[0] == self.image_size[0] and chess_img.shape[0] == self.image_size[0]), \
            "Image size does not match the given value {}.".format(self.image_size)
        # to gray
        gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        # find chessboard corners
        ret, img_corners = cv2.findChessboardCorners(gray, (corner_width, corner_height))

        # If found, add object points, image points (after refining them).
        if ret:
            objs_corner.append(obj_corner)
            img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(square_size // 2, square_size // 2),
                                           zeroZone=(-1, -1), criteria=criteria)
            imgs_corner.append(img_corners)
            # import pdb; pdb.set_trace()

            # Draw and display the corners.
            cv2.drawChessboardCorners(gray, (corner_width, corner_height), img_corners, ret)
            if not os.path.exists('./chessboard'):
                os.makedirs('./chessboard')
            cv2.imwrite(os.path.join('./chessboard', img_path.split('/')[-1]), gray)
        else:
            print("Fail to find corners in {}.".format(img_path))

        # calibration
        ret, self.matrix, self.dist, rvecs, tveces = cv2.calibrateCamera(objs_corner, imgs_corner, gray.shape[::-1],
                                                                             None, None)
        self.rotation_mtx, _ = cv2.Rodrigues(rvecs[0])
        self.translation_mtx = tveces[0]

        return [rvecs[0], tveces[0]]


    def rectify_image(self, img):
        """
        undistortion
        :param img:
        :return:
        """
        self.load_params()
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv2.undistort(img, self.matrix, self.dist, None, self.new_camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv2.resize(dst, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)
        return dst

    def get_extrinsic_params(self, chess_img_path='../datas/carlibration/color/chessboard_final.jpg',
                             corner_nums=[9, 6], square_size=25,
                             save_path_dir='../temps/chessboards/'
                             ):
        """
        获取相机外部参数, 需要新的标定图片
        :return:
        """
        self.load_params()

        if not os.path.exists(chess_img_path):
            print("Please check the image path")
            raise ValueError

        img = cv2.imread(chess_img_path)
        corner_height, corner_width = corner_nums

        obj_corner = np.zeros([corner_height * corner_width, 3], dtype=np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)  # (w*h)*2

        # 设置世界坐标的坐标
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        # 设置角点查找限制
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 查找角点
        ret, img_corners = cv2.findChessboardCorners(gray, (corner_height, corner_width), )
        print("Obtained corners num:", len(img_corners))
        print("ret", ret)

        # If found, add object points, image points (after refining them).
        objs_corner = []  # 3d point in real world space.
        imgs_corner = []  # 2d point in image plane.
        if ret:
            objs_corner.append(obj_corner)
            exact_corners = cv2.cornerSubPix(gray, img_corners, winSize=(square_size // 2, square_size // 2),
                                           zeroZone=(-1, -1), criteria=criteria)

            # Draw and display the corners.
            cv2.drawChessboardCorners(gray, (corner_width, corner_height), exact_corners, ret)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            cv2.imwrite(os.path.join(save_path_dir, chess_img_path.split('/')[-1]), gray)
        else:
            print("Fail to find corners in {}.".format(chess_img_path))

        # 设置(生成)标定图在世界坐标中的坐标
        world_points = np.zeros([corner_height * corner_width, 3], dtype=np.float32)
        world_points[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)  # (w*h)*2

        # 设置世界坐标的坐标
        axis = np.float32([[25*1, 0, 0], [0, 0, 0], [25*1, 25*8, 0]]).reshape(-1, 3)

        print("利用carlibration所得的旋转参量、平移参量")
        [rvec, tvec] = self.carlibrate_single(img_path=chess_img_path,
                                              corner_width=corner_width,
                                              corner_height=corner_height,
                                              square_size=square_size)
        print('carlibration() 函数所求, 旋转参量: ', rvec)
        print('carlibration() 函数所求, 平移参量: ', tvec)
        print("内部参数矩阵:\n", self.matrix)

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, self.matrix, self.dist)
        __tested_imgpts = imgpts[:]
        print("测试转化后 二维坐标\n", imgpts)

        rotation_mtx, jacb = cv2.Rodrigues(rvec)
        rotation_mtx_I = np.linalg.inv(rotation_mtx)
        # 可视化角点
        # print(img_corners)
        _img = self.draw(img, img_corners, imgpts)
        # cv2.imshow('img', img)
        cv2.imwrite(os.path.join(save_path_dir, 'projection_img.jpg'), _img)


        print("内部参数矩阵:\n", self.matrix)

        test_world_point = self.image2world(img_point=__tested_imgpts[0][0],
                                            rotation_mtx=self.rotation_mtx,
                                            translation_mtx=self.translation_mtx,
                                            camera_mtx=self.matrix,
                                            z_constS=500)
        test_world_point1 = self.image2world(img_point=__tested_imgpts[1][0],
                                            rotation_mtx=self.rotation_mtx,
                                            translation_mtx=self.translation_mtx,
                                            camera_mtx=self.matrix,
                                             z_constS=500)
        test_world_point2 = self.image2world(img_point=__tested_imgpts[2][0],
                                            rotation_mtx=self.rotation_mtx,
                                            translation_mtx=self.translation_mtx,
                                            camera_mtx=self.matrix,
                                             z_constS=500)
        print("测试三维坐标:\n", test_world_point[:, 0], test_world_point1[:, 0], test_world_point2[:, 0])
        imgpt, jac = cv2.projectPoints(test_world_point, rvec, tvec, self.matrix, self.dist)
        print("相应投影坐标:", imgpt)

        chess_corner_coords = [[0, 0], [0, 1], [1, 0], [0, 5], [8, 0],
                               [8, 5], [3, 3], [7, 2], [6, 3], [5, 4],
                               [7, 5], [8, 3], [4, 4]]
        chess_corner_coords = np.array(chess_corner_coords)



        # temp = chess_corner_coords[0, :]
        # chess_corner_coords[0,:] = chess_corner_coords[1,:]
        # chess_corner_coords[0,:] = temp
        idx_list = []
        for item in chess_corner_coords:
            idx_list.append([item[1], item[0]])
        idx_list = np.array(idx_list)
        idxs = []
        for item in idx_list:
            idxs.append(9 * (5 - item[0]) + item[1])
        idxs = np.array(idxs)

        # slected img pixel coordiantes
        selected_corners = img_corners[idxs]
        print("selected corners:\n", selected_corners)
        selected_wcp = []
        print("\nbegin to convert:")
        for sc in selected_corners:
            print(sc[0])
            temps = self.image2world(img_point=sc[0],
                                     rotation_mtx=rotation_mtx,
                                     translation_mtx=tvec,
                                     camera_mtx=self.matrix,
                                     z_constS=500
                                     )
            selected_wcp.append(temps)
        print(selected_wcp)
        selected_wcp = np.array(selected_wcp)[:, :2, 0]
        print(selected_wcp.shape)
        print(selected_wcp)

        relative_wcp = selected_wcp - selected_wcp[0]
        print("relative wpc:\n", relative_wcp * 500)
        relative_wcp[:, 0] = relative_wcp[:, 0] / relative_wcp[1][0]
        relative_wcp[:, 1] = relative_wcp[:, 1] / relative_wcp[2][1]
        print(relative_wcp)
        # TODO: 少了一个系数

        # tested_corner_1 = [img_corners[item[0]] ]

        # p1 = self.image2world(img_point=[524.03156, 239.6996],
        #                       rotation_mtx=self.rotation_mtx,
        #                       translation_mtx=self.translation_mtx,
        #                       camera_mtx=self.matrix)
        # p2 = self.image2world(img_point=[425.8834,  231.11479],
        #                       rotation_mtx=self.rotation_mtx,
        #                       translation_mtx=self.translation_mtx,
        #                       camera_mtx=self.matrix)
        # p3 = self.image2world(img_point=[491.0921,  209.83522],
        #                       rotation_mtx=self.rotation_mtx,
        #                       translation_mtx=self.translation_mtx,
        #                       camera_mtx=self.matrix)
        # p1, jac = cv2.projectPoints(p1, rvec, tvec, self.matrix, self.dist)
        # p2, jac = cv2.projectPoints(p2, rvec, tvec, self.matrix, self.dist)
        # p3, jac = cv2.projectPoints(p3, rvec, tvec, self.matrix, self.dist)
        # print(p1, p2, p3)


        # wc_point = self._image2world_pnp(image_point=[524.03156, 239.6996],
        #                                  rotation_mtx=self.rotation_mtx,
        #                                  translation_mtx=self.translation_mtx,
        #                                  camera_mtx=self.matrix)
        # print("test PNP coordinates:", wc_point)
        # imgpt, jac = cv2.projectPoints([wc_point], rvec, tvec, self.matrix, self.dist)
        # print("相应投影坐标:", imgpt)


        # test pnp method
        print(self.rotation_mtx.shape, self.translation_mtx.shape)
        wcp = self._image2world_pnp(image_point=[302.23053, 245.11359, 0],
                                    rotation_mtx=self.rotation_mtx, translation_mtx=self.translation_mtx,
                                    zConst_W=50, zConst_C=None)
        print("Using PNP method from GAN:\n", wcp)

    def _extrinsic_pnp(self, chess_img_path='../datas/carlibration/color/chessboard_final.jpg',
                       ):
        """
        using a chessboard image to calculate the extrinsic parameters:
            rotation matrix;
            translation matrix;
            scale factor
        :param chess_img_path:
        :return:
        """
        pass


    def image2world(self, img_point, rotation_mtx, translation_mtx, camera_mtx, z_constS):
        """
        convert image(pixel) coordinates to world (X, Y, .) coordinates
        TODO: add undistortion-bias to the transformation
        :param img_point:       list or np.array([u, v])
        :param rotation_mtx:    extrinsic parameters, rotation matrix
        :param translation_mtx:
        :param camera_mtx:      intrinsic parameters,
        :param z_constS:        when convert to 3D points, it need a depth params at Z axis.
        :return:
        """
        if len(img_point) == 2:
            _img_point = np.array([[img_point[0], img_point[1], 1]])
        elif len(img_point) == 3:
            _img_point = np.array([img_point])
        else:
            print("please check the image coordinates")
            raise ValueError
        # print("image points", img_point, _img_point, _img_point.T.shape)

        rotation_mtx_I = np.linalg.inv(rotation_mtx)
        camera_mtx_I = np.linalg.inv(camera_mtx)
        translation_mtx = np.array(translation_mtx)
        # print(camera_mtx_I.shape)
        # print(translation_mtx.shape)

        temp_mtx = np.dot(camera_mtx_I, _img_point.T) * z_constS - translation_mtx
        # print(temp_mtx.shape)

        world_3d = rotation_mtx_I.dot(temp_mtx)

        return world_3d

    def _image2world_pnp(self, image_point, rotation_mtx, translation_mtx, zConst_W=None, zConst_C=None):
        """
        image_point: np.array([[u], [v]], dtype=np.float)
        zConst_W: depth in world coordinate system.
        solve the scale factor s / zConst_C.
        """
        assert (not zConst_W is None) or (not zConst_C is None)
        # 3 x 3 rotation matrix
        revc_M1 = rotation_mtx
        if zConst_C is None:
            tempMat1 = np.linalg.inv(revc_M1).dot(np.linalg.inv(self.matrix)).dot(image_point)
            tempMat2 = np.linalg.inv(revc_M1).dot(translation_mtx)
            print(tempMat1, '\n',  tempMat2)
            s = (zConst_W + tempMat2[2][0]) / tempMat1[2]
            print("r_cev-1 * t_vec[2, 0]:{}".format(tempMat2[2, 0]))
        else:
            s = zConst_C
        print('scales:', s)
        _image_point = np.array([image_point[:]])
        x = (_image_point[0, 0] - self.matrix[0, 2]) / self.matrix[0, 0]
        y = (_image_point[0, 1] - self.matrix[1, 2]) / self.matrix[1, 1]
        r2 = x * x + y * y
        k1, k2, k3, p1, p2 = self.dist[0]
        x = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        x = x * self.matrix[0, 0] + self.matrix[0, 2]
        y = y * self.matrix[1, 1] + self.matrix[1, 2]
        image_point = np.array([[x], [y], [1]])
        wc_point = np.dot(np.linalg.inv(revc_M1), (np.linalg.inv(self.matrix).dot(image_point) * s - translation_mtx))
        return wc_point



    def draw(self, img, corners, imgpts):
        print("image points:\n", imgpts)

        corner = tuple(corners[8].ravel())      # 选择以第几个corner为起点?
        print("length of the corners {}".format(len(corners)))
        print("corner points:", corner)
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
        return img


def coornidinates_transformer(p_point, inner_mtx, r_mtx, t_mtx):
    """
    convert world point coordinates to pixel_point coordinates
    :param w_point:
    :param p_point:
    :return:
    """
    inner_mtx = np.matrix(inner_mtx)
    inner_mtx_I = inner_mtx.I
    print(inner_mtx)
    print(inner_mtx_I)
    pass

def  calcu_offset(point1, point2):
    assert len(point1) == len(point2)

    vec = np.array(point1) - np.array(point2)
    return vec



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=str, default='480x640', help='height*width of image')
    parser.add_argument('--mode', type=str, choices=['calibrate', 'rectify_image'], default='rectify_image',
                        help='to calibrate or rectify')
    parser.add_argument('--square', default=25, type=int, help='size of chessboard square, by mm')
    parser.add_argument('--corner', type=str, default='9x6', help='height*width of chessboard corner')
    parser.add_argument('--video_path', type=str, help='video to rectify')
    parser.add_argument('--image_data', type=str, default='')
    parser.add_argument('--root', type=str, default='./chess')
    parser.add_argument('--camera_id', type=int, help='camera_id, default=0', default=0)
    args = parser.parse_args()
    calibrator = None
    # python calibration_realsense.py --image_size 480x640 --mode calibrate --corner 9x6 --square 25
    # python calibration_realsense.py --image_size 480x640 --mode rectify_image --camera_id 0
    try:
        image_size = tuple(int(i) for i in args.image_size.split('x'))
        calibrator = CameraCalibrator(image_size, chessboards_dir='../datas/carlibration/color/')
    except:
        print("Invalid/Missing parameter: --image_size. Sample: \n\n"
              "    --image_size 1920*1080\n")
        exit(-1)

    if args.mode == 'calibrate':
        if not args.corner or not args.square:
            print("Missing parameters of corner/square. Using: \n\n"
                  "    --corner <width>x<height>\n\n"
                  "    --square <length of square>\n")
            exit(-1)
        corner = tuple(int(i) for i in args.corner.split('x'))
        if calibrator.calibration(corner[1], corner[0], args.square):
            calibrator.save_params()
        else:
            print("Calibration failed.")
    elif args.mode == 'rectify_image':
        # df = pd.read_csv(ars.image_data)
        # for index, path in df['filename']:
        #     img = cv2.imread(os.path.join(args.root, path))
        #     img_dst = calibrator.rectify_image(img)
        #     cv2.imwrite('undistort_'+path, img_dst)
        file_names = glob.glob('../datas/carlibration/color/*.jpg') + glob.glob('../datas/carlibration/color/*.jpg') + \
                     glob.glob('.../datas/carlibration/color/*.png')


        if not os.path.exists('../datas/undistort_chess'):
            os.makedirs('../datas/undistort_chess')
        for file in file_names:
            img = cv2.imread(file)
            print(file)
            print(img.shape)
            # import pdb; pdb.set_trace()
            img_dst = calibrator.rectify_image(img)
            cv2.imwrite(os.path.join('../datas/undistort_chess', 'undistort_' + file.split('/')[-1]), img_dst)


    else:
        print("Invalid/Missing parameter '--mode'. Please choose from ['calibrate', 'rectify'].")
        exit(-1)

    print("\n****************************\n")
    print("try to visualize the co-ordinates")
    calibrator.get_extrinsic_params(chess_img_path='../datas/carlibration/color/chessboard_final.jpg')


    # cammer_mtx = calibrator.matrix
    # # coornidinates_transformer(p_point=[524.03156, 239.6996], inner_mtx=cammer_mtx,
    # #                           r_mtx=None, t_mtx=None)
    #
    #
    # # 获取所采集点的三维坐标
    # df = pd.read_csv('../datas/标定.csv')
    # df_data = np.array(df)
    # df_data = df_data[:, -4:-1] * 1000  # from m to mm
    # # print(df_data[:5])
    # origin = df_data[0]
    # relative_data = df_data - df_data[0]
    # # print(relative_data)
    #
    # chess_corners_wcp = relative_data[:13]
    # centri_lid_wcp = relative_data[13]
    # centri_center_wcp = relative_data[14]
    # tubes_4_wcp = relative_data[15:]
    # np.save('../datas/tested_corner_wcp.npy', chess_corners_wcp)
    #
    # #
    # chess_corner_coords = [[0, 0], [0, 1], [1, 0], [0, 5], [8, 0],
    #                        [8, 5], [3, 3], [7, 2], [6, 3], [5, 4],
    #                        [7, 5], [8, 3], [4, 4]]
    # idx_list = []
    # for item in chess_corner_coords:
    #     idx_list.append([item[1], item[0]])
    # idx_list = np.array(idx_list)



