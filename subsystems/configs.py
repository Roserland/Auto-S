import os

class centrifuge():
    def __init__(self):
        self.center_pos = (390, 240)

        self.w = 640
        self.h = 480
        self.crop_size = 425
        self.tube_bgr_thres = [(0, 120), (75, 150), (160, 255)]

    def check(self):
        pass


class socketServer():
    def __init__(self):
        self.ip = '192.168.1.103'
        self.port = 23333
