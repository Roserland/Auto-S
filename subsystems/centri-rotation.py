# 根据四个marker，将离心机图片旋转某个角度，从而估算出孔洞
# 同时剔除无关背景，便于识别
#

import numpy as np
import cv2
import os, yaml, time, json
import pandas as pd