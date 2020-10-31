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

    def _draw(self, img, obj_corners, img_corners):
        corners = np.vstack([obj_corners[0], obj_corners[1], obj_corners[9], obj_corners[10]])
        corner_center = obj_corners[0]
        corner_z = np.array([corner_center[0], corner_center[1], corner_center[2] + 96])
        corner_x = np.array([corner_center[0] + 48, corner_center[1], corner_center[2]])
        corner_y = np.array([corner_center[0], corner_center[1] + 48, corner_center[2]])
        corners = np.vstack([corner_center, corner_x, corner_y, corner_z])
        imgpts, jac = cv2.projectPoints(corners, self.rvec, self.tvec, self.matrix, self.dist)

        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()), tuple(imgpts[1].astype(np.int).ravel()),
                       (255, 0, 0), 5)
        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()), tuple(imgpts[2].astype(np.int).ravel()),
                       (0, 255, 0), 5)
        img = cv2.line(img, tuple(imgpts[0].astype(np.int).ravel()),
                       tuple(imgpts[3].astype(np.int).astype(np.int).ravel()), (0, 0, 255), 5)
        cv2.imwrite('coordinate.jpg', img)
        return img

    def calibration(self, file_path=None, image_points=None, depth=None, world_points=None, corner_width: int = 9,
                    corner_height: int = 6, square_size: float = 24):
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

    def _calibrate_camera_intrinsic(self, file_path, obj_corners, \
                                    corner_width=9, corner_height=6, square_size=24, criteria=None):
        file_names = glob.glob(os.path.join(file_path, '*.JPG')) \
                     + glob.glob(os.path.join(file_path, '*.jpg')) \
                     + glob.glob(os.path.join(file_path, '*.png'))
        objs_corner = []  # 3d point in real world space.
        imgs_corner = []  # 2d point in image plane.

        for file_name in file_names:
            # read image
            chess_img = cv2.imread(file_name)
            assert (chess_img.shape[0] == self.image_size[0] and chess_img.shape[0] == self.image_size[0]), \
                "Image size does not match the given value {}.".format(self.image_size)
            # to gray
            gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv2.findChessboardCorners(gray, (corner_width, corner_height), None)

            # If found, add object points, image points (after refining them).
            if ret:
                objs_corner.append(obj_corners)
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
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, gray.shape[::-1], alpha=1,
                                                                    newImgSize=gray.shape[::-1])
        self.roi = np.array(roi)
        self._cal_re_projection_error(file_names, objs_corner, imgs_corner, rvecs, tveces)
        return ret

    def _cal_re_projection_error(self, filenames, objs_corner, imgs_corner, rvecs, tveces):
        total_error = 0
        for index in range(len(objs_corner)):
            img_corner2, _ = cv2.projectPoints(objs_corner[index], rvecs[index], tveces[index], \
                                               self.matrix, self.dist)
            error = cv2.norm(imgs_corner[index], img_corner2, cv2.NORM_L2) / len(img_corner2)
            total_error += error
            print(filenames[index], ': ', error)
        print("mean re-projection error: ", total_error / len(objs_corner))

    def _calibrate_camera_extrinsic(self, image_points=None, depth=None, world_points=None, \
                                    corner_width=9, corner_height=6, square_size=24, criteria=None):
        """
        word_poionts: the designed world coordinate system.
        """

        _, self.rvec, self.tvec, inliers = cv2.solvePnPRansac(obj_corners, img_corners, self.matrix, self.dist)
        # 获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues
        rotation_m, _ = cv2.Rodrigues(self.rvec)
        rotation_t = np.hstack([rotation_m, self.tvec])
        self.rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])

        self._draw(image, obj_corners, img_corners)

        return ret

    # 九点标定法
    def _cam2robotBase(self, cam_corners, obj_corners, corner_width=9, corner_height=6):
        # rotation matrix = rotation_t[0:3, 0:3], translate matrix = rotation_t[0:3, 3:]
        # Output vector indicating which points are inliers (1-inlier, 0-outlier).
        retval, rotation_t, inliers = cv2.estimateAffine3D(cam_corners, obj_corners)
        return rotation_t

    def _image2cam(self, image_point, zConst_C=0):
        """
        image_point: np.array([[u], [v]], dtppe=np.float)
        zConst_C: from depth image in camera coordinate system.
        """
        # Xc = (image_point[0, 0] - self.matrix[0, 2]) * zConst_C / self.matrix[0, 0]
        # Yc = (image_point[0, 1] - self.matrix[1, 2]) * zConst_C / self.matrix[1, 1]
        # return np.array([[Xc], [Yc], [zConst_C]], dtype=np.float)
        image_point = np.vstack([image_point.transpose(1, 0), np.array([[1]])]).astype(np.float32)
        return zConst_C * np.dot(np.linalg.inv(self.matrix), image_point)

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
            tempMat1 = np.linalg.inv(revc_M1).dot(np.linalg.inv(self.matrix)).dot(image_point)
            tempMat2 = np.linalg.inv(revc_M1).dot(self.tvec)
            s = (zConst_W + tempMat2[2, 0]) / tempMat1[2, 0]
        else:
            s = zConst_C
        x = (image_point[0, 0] - self.matrix[0, 2]) / self.matrix[0, 0]
        y = (image_point[0, 1] - self.matrix[1, 2]) / self.matrix[1, 1]
        r2 = x * x + y * y
        k1, k2, k3, p1, p2 = self.dist[0]
        x = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        x = x * self.matrix[0, 0] + self.matrix[0, 2]
        y = y * self.matrix[1, 1] + self.matrix[1, 2]
        image_point = np.array([[x], [y], [1]])
        wc_point = np.dot(np.linalg.inv(revc_M1), (np.linalg.inv(self.matrix).dot(image_point) * s - self.tvec))
        return wc_point

    def _image2world_affine(self, image_point=None, zConst_C=None):
        image_point = np.array([[image_point[0, 0]], [image_point[0, 1]], [1]]).astype(np.float)
        revc_M1 = self.cam2base_affine[0:3, 0:3]
        T = self.cam2base_affine[0:3, 3:]
        wc_point = np.linalg.inv(revc_M1).dot(np.linalg.inv(self.matrix)).dot(image_point) * zConst_C - np.linalg.inv(
            revc_M1).dot(T)
        return wc_point

    def rectify_image(self, img):
        self.load_params()
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv2.undistort(img, self.matrix, self.dist, None, self.new_camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv2.resize(dst, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)
        return dst

    def inpaint(self, img, missing_value=0):
        """
        Inpaint missing values in depth image. [H, W]
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = np.expand_dims(img, axis=2)
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
        # img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=str, default='480x640', help='height * width of image')
    parser.add_argument('--square', default=25, type=int, help='size of chessboard square, by mm')
    parser.add_argument('--corner', type=str, default='9x6', help='height * width of chessboard corner')
    parser.add_argument('--image_data', type=str, default='../datas/carlibration/color')
    args = parser.parse_args()
    calibrator = None

    image_size = tuple(int(i) for i in args.image_size.split('x'))
    corner = tuple(int(i) for i in args.corner.split('x'))

    ## prepare data to compute camera extrinsic.
    image = cv2.imread('../datas/carlibration/color/chessboard_final.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df = pd.read_csv('../datas/wrold_coords.csv', header=None)
    obj_corners = np.array(df.values[:, 12:15].astype(np.float) * 1000)
    ## be careful about the corresponding relationship between A to B.
    obj_corners = obj_corners[::-1, :]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, img_corners = cv2.findChessboardCorners(gray, (corner[1], corner[0]), None)
    img_corners = cv2.cornerSubPix(gray, img_corners, winSize=(args.square // 2, args.square // 2),
                                   zeroZone=(-1, -1), criteria=criteria)
    print("img_corners-0:\n", img_corners)
    img_corners = np.concatenate([img_corners, np.expand_dims(np.array(
        [[256.81818181818176, 294.45454545454544],
         [380.4545454545454, 296.27272727272725],
         [498.6363636363636, 299.9090909090909],
         [671.3636363636363, 303.5454545454545],
         [833.181818181818, 303.5454545454545],
         [1005.9090909090908, 332.6363636363636],
         [335.0, 592.6363636363636],
         [442.27272727272725, 590.8181818181818],
         [551.3636363636363, 589.0],
         [703.0, 596.0],
         [855.0, 598.0],
         [1004.0, 598.0],
         [377.0, 774.0],
         [479.0, 776.0],
         [580.0, 777.0],
         [722.0, 779.0],
         [866.0, 781.0],
         [1007.0, 783.0],
         [896.0, 959.0],
         [1010.0, 959.0]
         ]), axis=1)], axis=0)

    obj_corners = np.vstack([obj_corners, np.array([
        [0.032956400678150505, 0.35997132749858896, 0.09924961732985867],
        [0.03156121336846241, 0.4021895730740082, 0.10027157808220943],
        [0.03234809686177748, 0.4423294447285742, 0.1016863875717318],
        [0.032194857100136656, 0.49873178517377353, 0.10393247634180006],
        [0.03142117735197521, 0.5564165917061202, 0.10524772927841243],
        [0.039359220454285464, 0.6131017755945851, 0.10302939783714267],
        [0.12371266193855365, 0.3677524147888837, 0.07200257982862376],
        [0.12314415955709872, 0.4037593342137452, 0.07190126371905095],
        [0.12412207501883374, 0.4466758505692857, 0.07280393209716934],
        [0.1233765047165455, 0.5003429508462289, 0.07283232308095842],
        [0.1254451326941377, 0.5556132552182049, 0.07261541079441232],
        [0.12447542531090977, 0.6129100654517537, 0.07476476902364275],
        [0.19266993117957965, 0.3659864505773379, 0.046394835977228085],
        [0.19333610823348374, 0.40471068114588815, 0.04682424338337085],
        [0.1933988054746717, 0.4454039723268175, 0.04774546664383661],
        [0.1918604899078508, 0.5019360489494338, 0.04948457168873256],
        [0.1924959559755324, 0.5571173073396807, 0.04947814983694146],
        [0.19199583246421004, 0.6139128636004798, 0.050498533829356124],
        [0.2691298885442751, 0.5661667115758835, 0.021884357041168256],
        [0.2676372218965072, 0.6178722607862462, 0.022472336498830445]]) * 1000])

    calibrator = CameraCalibrator(image_size, load=False, param_file='camera_params.yaml', \
                                  image_data=args.image_data, image_points=img_corners, depth=None,
                                  world_points=obj_corners, \
                                  corner_width=corner[1], corner_height=corner[0], square_size=args.square
                                  )

    # Test camera instric / extrinsic.
    depth = np.load('00018.npy')
    image = cv2.imread('00018.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image_points = cv2.findChessboardCorners(gray, (9, 6), None)
    world_points = []
    cam_points = []
    for image_point in image_points:
        world_point = calibrator._image2world_pnp(image_point=image_point,
                                                  zConst_C=depth[int(image_point[0, 1]), int(image_point[0, 0])])
        # world_point = calibrator._image2world_affine(image_point=image_point, zConst_C=depth[int(image_point[0,1]), int(image_point[0,0])])
        world_points.append(world_point)
    print(world_points)

    depth = np.load('exp.npy')
    depth = calibrator.inpaint(depth)
    image = cv2.imread('exp.jpg')
    image_points = np.expand_dims(np.array(
        [[108.9090909090909, 854.9090909090909],
         [163.45454545454544, 427.6363636363636],
         [1521.6363636363635, 800.3636363636363],
         [1719.8181818181818, 413.09090909090907],
         [1961.6363636363635, 800.3636363636363]]), axis=1)
    world_points = []
    for image_point in image_points:
        world_point = calibrator._image2world_pnp(image_point=image_point,
                                                  zConst_C=depth[int(image_point[0, 1]), int(image_point[0, 0])])
        # world_point = calibrator._image2world_affine(image_point=image_point, zConst_C=depth[int(image_point[0,1]), int(image_point[0,0])])
        world_points.append(world_point)
    print(world_points)
    print(calibrator._world2image(
        world_point=np.array([[215.13058890275177], [347.8963519218469], [128.21091611040472]])))