import cv2
import numpy as np
import os

def show_a_img(img, name='test1', window_size=(10,10)):
    """
    simplely show a img
    :param img:
    :param window_size:
    :return:
    """
    cv2.namedWindow(name, 1)
    cv2.resizeWindow(name, window_size[0], window_size[1])
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    cv2.imwrite('./temps/'+name+'.jpg', img)

def find_max_contours(bi_img, src_img, _save_path='./temps/__with_rect.jpg'):
    """
    寻找最大连通域
    :param img:
    :return:
    """
    _bi_img = bi_img[:, :]
    print(_bi_img.shape)

    contours, hierarchy = cv2.findContours(_bi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours num is {}".format(len(contours)))
    area = []


    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    if len(area) == 0:
        print('***************-----**************')
        print()
        return
    else:
        print('OK')
    max_idx = np.argmax(area)
    print("max area num is {}".format(area[max_idx]))

    # 绘制轮廓
    cv2.drawContours(src_img, contours, max_idx, color=(3, 3, 3), thickness=2)
    show_a_img(src_img)

    # 绘制最小矩形框
    ## 定位矩形框
    p1, p2 = find_min_rec(img=src_img, contours=contours, cont_idx=max_idx)
    print(p1, p2)
    ## draw
    draw_rectangle(img=src_img, pt1=p1, pt2=p2, save_path=_save_path)

    return contours, max_idx, (p1, p2)

def find_min_rec(img, contours, cont_idx, ):
    """
    找到含括目标的最小矩形框
    :param img:
    :param contours: 轮廓列表， list
    :param cont_idx: 对应轮廓列表中的索引
    :return: (left-top-point, right-bottom-point)
    """
    minRect = cv2.minAreaRect(contours[cont_idx])
    print("minRect is {}".format(minRect))
    x, y, w, h = cv2.boundingRect(contours[cont_idx])
    p1 = (x, y)
    p2 = (x+w, y+h)

    return p1, p2

def draw_rectangle(img, pt1, pt2, save_path='./temps/__with_rect.jpg'):
    """
    在某图中，画出含括目标的最小矩形框
    :param img: 原图
    :param pt1: 矩形框 左上角点
    :param pt2: 矩形框 右下角点
    :return:
    """
    cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(183, 152, 63), thickness=3)
    s = cv2.imwrite(save_path, img)

    if not s:
        print('Not saved')
        print("Please check if the path is right or if the directory exists")
        raise ValueError

def get_image_path(data_dir):
    file_list = os.listdir(data_dir)
    final = []
    for item in file_list:
        if '.jpg' in item:
            final.append(item)
        else:
            pass
    return final

def rack_unit_process(img_path, dst_path):
    img = cv2.imread(img_path)

    thres = thres_rack(img=img, r_thres=(120, 190), g_thres=(40,  80), b_thres=(20,  65))
    _, thres = cv2.threshold(thres, 127, 255, cv2.THRESH_BINARY)

    find_max_contours(bi_img=thres, src_img=img, _save_path=dst_path)

def main():
    print(123)
    imgPath = './figures/color0.jpg'

    img = cv2.imread(imgPath)
    r_low, r_high = (120, 190)
    g_low, g_high = (30,  60)
    b_low, b_high = (20,  65)

    b, g, r = cv2.split(img)
    thres = thres_rack(img=img, r_thres=(120, 190), g_thres=(40,  80), b_thres=(20,  65))
    _, thres = cv2.threshold(thres, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    thres = cv2.dilate(src=thres, kernel=kernel)
    print(type(thres))

    cv2.imwrite('./figures/thres-temp-100-200.jpg', thres)
    print(thres.shape)
    # find_max_contours(bi_img=thres, src_img=img)

    # ---------------------------
    # Racks Large
    # rack_large_dir = get_image_path(data_dir='./test images/tube-rack-large')
    # save_dir = './result_images/tube-rack-large'
    # for i in range(len(rack_large_dir)):
    #     img_name = rack_large_dir[i]
    #     src_img_path = os.path.join('./test images/tube-rack-large', img_name)
    #     dst_img_path = os.path.join(save_dir, 'rectLines-'+ img_name)
    #
    #     rack_unit_process(img_path=src_img_path, dst_path=dst_img_path)

    # Racks Small
    original_dir_name = '/centrifuge-bottle/'
    rack_small_dir = get_image_path(data_dir='./test images/'+original_dir_name)
    save_dir = './result_images/' + original_dir_name
    for i in range(len(rack_small_dir)):
        img_name = rack_small_dir[i]
        src_img_path = os.path.join('./test images/'+original_dir_name, img_name)
        dst_img_path = os.path.join(save_dir, 'rectLines-'+ img_name)

        rack_unit_process(img_path=src_img_path, dst_path=dst_img_path)




if __name__ == '__main__':
    main()