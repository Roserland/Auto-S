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
    def __init__(self, image_size:tuple):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float)
        self.new_camera_matrix = np.zeros((3, 3), np.float)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int)


    def load_params(self, param_file:str='camera_params.yaml'):
        if not os.path.exists(param_file):
            print("File {} does not exist.",format(param_file))
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
            for column in range(self.new_camera_matrix.shape[1])] for row in range(self.new_camera_matrix.shape[0])]
        dist = [float(self.dist[0][i]) for i in range(self.dist.shape[1])]
        roi = [int(self.roi[i]) for i in range(self.roi.shape[0])]
        height = self.image_size[0]
        width = self.image_size[1]
        records = {
            'image height': height,
            'image width': width,
            'camera_matrix': mat,
            'new_camera_matrix': new_mat,
            'camera_distortion': dist,
            'roi': roi
        }
        with open(save_path, 'w') as f:
            yaml.dump(records, f)
        print("Saved params in {}.".format(save_path))


    # corner_width = 9, corner_height = 6;
    def _cal_real_corner(self, corner_width, corner_height, square_size):
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)  # (w*h)*2
        return obj_corner * square_size


    def calibration(self, corner_width:int, corner_height:int, square_size:float):
        file_names = glob.glob('./chess/*.JPG') + glob.glob('./chess/*.jpg') + glob.glob('./chess/*.png')
        objs_corner = []    # 3d point in real world space.
        imgs_corner = []    # 2d point in image plane.
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数 30 和最大误差容限 0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 获取标定板角点的位置
        obj_corner = self._cal_real_corner(corner_width, corner_height, square_size)
        
        for file_name in file_names:
            # read image
            chess_img = cv2.imread(file_name)
            assert (chess_img.shape[0] == self.image_size[0] and chess_img.shape[0] == self.image_size[0]), \
                "Image size does not match the given value {}.".format(self.image_size)
            # to gray
            gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv2.findChessboardCorners(gray, (corner_width, corner_height))

            # If found, add object points, image points (after refining them).
            if ret:
                objs_corner.append(obj_corner)
                img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(square_size//2, square_size//2),
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
        ret, self.matrix, self.dist, rvecs, tveces = cv2.calibrateCamera(objs_corner, imgs_corner, gray.shape[::-1], None, None)
        # matrix, 内参数矩阵 （3 * 3）
        # dist, 畸变系数（k1,k1,k3,p1,p2)
        # revecs 旋转向量， 外参数
        # tveces 平移向量， 外参数
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, gray.shape[::-1], alpha=0, newImgSize=gray.shape[::-1])
        self.roi = np.array(roi)
        
        # 计算重投影误差
        total_error = 0
        for index in range(len(objs_corner)):
            img_corner2, _ = cv2.projectPoints(objs_corner[index], rvecs[index], tveces[index],\
                 self.matrix, self.dist)
            error = cv2.norm(imgs_corner[index],img_corner2, cv2.NORM_L2) / len(img_corner2)
            total_error += error
            print(file_name, ': ', error)
        print ("mean projection error: ", total_error/len(objs_corner))
        return ret


    def rectify_image(self, img):
        self.load_params()
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv2.undistort(img, self.matrix, self.dist, None, self.new_camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv2.resize(dst, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)
        return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type= str, default='3072x4096', help='height*width of image')
    parser.add_argument('--mode', type=str, choices=['calibrate', 'rectify_image'], default='rectify_image', help='to calibrate or rectify')
    parser.add_argument('--square', default=21, type=int, help='size of chessboard square, by mm')
    parser.add_argument('--corner', type=str, default='6x9', help='height*width of chessboard corner')
    parser.add_argument('--video_path', type=str, help='video to rectify')
    parser.add_argument('--image_data', type=str, default='')
    parser.add_argument('--root', type=str, default='./chess')
    parser.add_argument('--camera_id', type=int, help='camera_id, default=0', default=0)
    args = parser.parse_args()
    calibrator = None
    # python3 calibration.py --image_size 4096x3072 --mode calibrate --corner 9x6 --square 21
    # python3 calibration.py --image_size 4096x3072 --mode rectify_image --camera_id 0
    try:
        image_size = tuple(int(i) for i in args.image_size.split('x'))
        calibrator = CameraCalibrator(image_size)
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
        file_names = glob.glob('./chess/*.JPG') + glob.glob('./chess/*.jpg') + glob.glob('./chess/*.png')
        if not os.path.exists('./undistort_chess'):
                os.makedirs('./undistort_chess')
        for file in file_names:
            img = cv2.imread(file)
            print(file)
            # import pdb; pdb.set_trace()
            img_dst = calibrator.rectify_image(img)
            cv2.imwrite(os.path.join('./undistort_chess', 'undistort_'+file.split('/')[-1]), img_dst)


    else:
        print("Invalid/Missing parameter '--mode'. Please choose from ['calibrate', 'rectify'].")
        exit(-1)
