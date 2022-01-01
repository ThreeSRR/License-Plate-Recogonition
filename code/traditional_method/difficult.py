import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import template_matching, split_license, cv2ImgAddText


def task3_1(verbose=True):
    src = cv2.imread('../../resources/difficult/3-1.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b>120)*(r<60), np.ones_like(r), np.zeros_like(r))

    gray_mask = gray_image*mask
    gray_mask[gray_mask!=0]=255

    # 形态学处理，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary_mask = cv2.dilate(binary_mask, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,iterations = 2)

    # 获取车牌四点坐标，并计算矫正后四点坐标
    white = np.where(binary_mask == 255)
    y0, x0, y1, x1 = min(white[0]), min(white[1]), max(white[0]), max(white[1])

    deltax = x1 - x0
    deltay = white[0].shape[0] // deltax

    p1 = [x0, y1-deltay]
    p2 = [x1, y0]
    p3 = [x0, y1]
    p4 = [x1, y0+deltay]

    t1 = [x0, y1-deltay]
    t2 = [x0+int(1.5*deltax), y1-deltay]
    t3 = [x0, y1]
    t4 = [x0+int(1.5*deltax), y1]

    clip_license = src.copy()
    cv2.line(clip_license, p1, p2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p1, p3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p2, p4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p3, p4, color=(0, 0, 255), thickness=10)

    clip_license_trans = clip_license.copy()
    cv2.line(clip_license_trans, t1, t2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t1, t3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t2, t4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t3, t4, color=(0, 0, 255), thickness=10)

    # 四点透视矫正
    w, h, c = src.shape
    M = cv2.getPerspectiveTransform(np.array([p1, p2, p3, p4], dtype=np.float32), np.array([t1, t2, t3, t4], dtype=np.float32))

    src_fix = cv2.warpPerspective(src, M, (h, w))

    # 截取视角矫正后车牌
    license_img = src_fix.copy()
    license_img = license_img[t1[1]:t4[1],t1[0]:t4[0],:]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    cv2.line(src_fix, t1, t2, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t2, t4, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t4, t3, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t3, t1, color=(0, 0, 255), thickness=15)

    # 阈值分割，将车牌二值化
    ret, binary_license_img = cv2.threshold(gray_license_img, 130, 255, cv2.THRESH_BINARY)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 闭运算使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilate_lic_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_CLOSE, kernel,iterations = 1)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [clip_license_trans, src_fix, license_img]:
            plt.cla()
            plt.imshow(img[:,:,::-1])
            plt.show()

        plt.cla()
        for i,j in enumerate(word_images):  
            plt.subplot(1,8,i+1)
            plt.imshow(word_images[i],cmap='gray')
        plt.show()

    res_img = cv2ImgAddText(src, "".join(result), (0,0), textSize=350)
    plt.cla()
    plt.imshow(res_img)
    plt.show()


def task3_2(verbose=True):
    src = cv2.imread('../../resources/difficult/3-2.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b<170)*(r<165)*(g>175), np.ones_like(r), np.zeros_like(r))

    gray_mask = gray_image*mask
    gray_mask[gray_mask!=0]=255

    # 形态学处理，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel,iterations = 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,iterations = 2)

    # 获取车牌四点坐标，并计算矫正后四点坐标
    white = np.where(binary_mask == 255)
    y0, x0, y1, x1 = min(white[0]), min(white[1]), max(white[0]), max(white[1])

    deltax = x1 - x0
    deltay = white[0].shape[0] // deltax
    offset = 50

    p1 = [x0, y0-offset]
    p2 = [x1, y1-deltay-offset]
    p3 = [x0, y0+deltay]
    p4 = [x1, y1]

    t1 = [x0, y0-offset]
    t2 = [x0+int(1.5*deltax), y0-offset]
    t3 = [x0, y0+deltay]
    t4 = [x0+int(1.5*deltax), y0+deltay]

    clip_license = src.copy()
    cv2.line(clip_license, p1, p2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p1, p3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p2, p4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p3, p4, color=(0, 0, 255), thickness=10)

    clip_license_trans = clip_license.copy()
    cv2.line(clip_license_trans, t1, t2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t1, t3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t2, t4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t3, t4, color=(0, 0, 255), thickness=10)

    # 四点透视矫正
    w, h, c = src.shape
    M = cv2.getPerspectiveTransform(np.array([p1, p2, p3, p4], dtype=np.float32), np.array([t1, t2, t3, t4], dtype=np.float32))

    src_fix = cv2.warpPerspective(src, M, (h, w))

    # 截取视角矫正后车牌
    license_img = src_fix.copy()
    license_img = license_img[t1[1]:t4[1],t1[0]:t4[0],:]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    cv2.line(src_fix, t1, t2, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t2, t4, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t4, t3, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t3, t1, color=(0, 0, 255), thickness=15)

    # 阈值分割，将车牌二值化
    ret, binary_license_img = cv2.threshold(gray_license_img, 70, 255, cv2.THRESH_BINARY)

    binary_license_img[binary_license_img==255] = 1
    binary_license_img[binary_license_img==0] = 255
    binary_license_img[binary_license_img==1] = 0

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_CLOSE, kernel,iterations = 1)

    # 闭运算使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 10))
    dilate_lic_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_CLOSE, kernel,iterations = 1)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [clip_license_trans, src_fix, license_img]:
            plt.cla()
            plt.imshow(img[:,:,::-1])
            plt.show()

        plt.cla()
        for i,j in enumerate(word_images):  
            plt.subplot(1,8,i+1)
            plt.imshow(word_images[i],cmap='gray')
        plt.show()

    res_img = cv2ImgAddText(src, "".join(result), (0,0), textSize=350)
    plt.cla()
    plt.imshow(res_img)
    plt.show()


def task3_3(verbose=True):
    src = cv2.imread('../../resources/difficult/3-3.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b>150)*(r<70), np.ones_like(r), np.zeros_like(r))

    gray_mask = gray_image*mask
    gray_mask[gray_mask!=0]=255

    # 形态学处理，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,iterations = 3)

    # 获取车牌四点坐标，并计算矫正后四点坐标
    white = np.where(binary_mask == 255)
    y0, x0, y1, x1 = min(white[0]), min(white[1]), max(white[0]), max(white[1])

    deltax = x1 - x0
    deltay = white[0].shape[0] // deltax
    offset = 25

    p1 = [x0+offset, y1-deltay-offset]
    p2 = [x1, y0]
    p3 = [x0, y1]
    p4 = [x1-offset, y0+deltay]

    t1 = [x0, y1-deltay-offset]
    t2 = [x0+int(1.4*deltax), y1-deltay-offset]
    t3 = [x0, y1]
    t4 = [x0+int(1.4*deltax), y1]

    clip_license = src.copy()
    cv2.line(clip_license, p1, p2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p1, p3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p2, p4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license, p3, p4, color=(0, 0, 255), thickness=10)

    clip_license_trans = clip_license.copy()
    cv2.line(clip_license_trans, t1, t2, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t1, t3, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t2, t4, color=(0, 0, 255), thickness=10)
    cv2.line(clip_license_trans, t3, t4, color=(0, 0, 255), thickness=10)

    # 四点透视矫正
    w, h, c = src.shape
    M = cv2.getPerspectiveTransform(np.array([p1, p2, p3, p4], dtype=np.float32), np.array([t1, t2, t3, t4], dtype=np.float32))

    src_fix = cv2.warpPerspective(src, M, (h, w))

    # 截取视角矫正后车牌
    license_img = src_fix.copy()
    license_img = license_img[t1[1]:t4[1],t1[0]:t4[0],:]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    cv2.line(src_fix, t1, t2, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t2, t4, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t4, t3, color=(0, 0, 255), thickness=15)
    cv2.line(src_fix, t3, t1, color=(0, 0, 255), thickness=15)

    # 阈值分割，将车牌二值化
    ret, binary_license_img = cv2.threshold(gray_license_img, 130, 255, cv2.THRESH_BINARY)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 闭运算使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_lic_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_CLOSE, kernel,iterations = 3)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [clip_license_trans, src_fix, license_img]:
            plt.cla()
            plt.imshow(img[:,:,::-1])
            plt.show()

        plt.cla()
        for i,j in enumerate(word_images):  
            plt.subplot(1,8,i+1)
            plt.imshow(word_images[i],cmap='gray')
        plt.show()

    res_img = cv2ImgAddText(src, "".join(result), (0,0), textSize=350)
    plt.cla()
    plt.imshow(res_img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default="1", type=str, help='task id, choose from 1,2,3')
    parser.add_argument('--verbose', action='store_true', help='display result during process or not')


    args = parser.parse_args()
    task_id = args.task_id
    verbose = args.verbose

    task_func = {"1": task3_1, "2": task3_2, "3": task3_3}

    task_func[task_id](verbose=verbose)
