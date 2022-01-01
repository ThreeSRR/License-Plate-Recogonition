import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import template_matching, split_license, cv2ImgAddText


def task2_1(verbose=True):
    src = cv2.imread('../../resources/medium/2-1.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b>130)*(r<12), np.ones_like(r), np.zeros_like(r))

    gray_mask = gray_image * mask
    gray_mask[gray_mask!=0]=255

    # 闭运算，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    license_area = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel,iterations = 2)

    white_area = np.where(license_area==255)

    # 找到车牌区域边界，分割车牌
    x0, y0, x1, y1 = min(white_area[1]), min(white_area[0]), max(white_area[1]), max(white_area[0])

    clip_license = src.copy()
    cv2.rectangle(clip_license, (x0, y0), (x1, y1), (0, 0, 255), 10)

    license_img = src[y0:y1,x0:x1]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 膨胀使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilate_lic_img = cv2.dilate(binary_license_img, kernel)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [gray_mask, license_area]:
            plt.cla()
            plt.imshow(img, cmap='gray')
            plt.show()

        plt.cla()
        plt.imshow(license_img[:,:,::-1])
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


def task2_2(verbose=True):
    src = cv2.imread('../../resources/medium/2-2.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b>130)*(r<11), np.ones_like(r), np.zeros_like(r))
    gray_mask = gray_image * mask
    gray_mask[gray_mask!=0]=255

    # 闭运算，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    license_area = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel,iterations = 2)

    white_area = np.where(license_area==255)

    # 找到车牌区域边界，分割车牌
    x0, y0, x1, y1 = min(white_area[1]), min(white_area[0]), max(white_area[1]), max(white_area[0])

    clip_license = src.copy()
    cv2.rectangle(clip_license, (x0, y0), (x1, y1), (0, 0, 255), 10)

    license_img = src[y0:y1,x0:x1]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)
    
    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel, iterations = 1)

    # 膨胀使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    dilate_lic_img = cv2.dilate(binary_license_img, kernel)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [gray_license_img, binary_license_img, dilate_lic_img]:
            plt.cla()
            plt.imshow(img, cmap='gray')
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
    src = cv2.imread('../../resources/medium/2-3.jpg')
    gray_image = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 分离RGB通道，阈值分割
    b, g, r = cv2.split(src)
    mask = np.where((b>100)*(r<80), np.ones_like(r), np.zeros_like(r))
    gray_mask = gray_image * mask
    gray_mask[gray_mask!=0]=255

    # 闭运算，获得完整车牌区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    license_area = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel,iterations = 2)

    white_area = np.where(license_area==255)

    # 找到车牌区域边界，分割车牌
    x0, y0, x1, y1 = min(white_area[1]), min(white_area[0]), max(white_area[1]), max(white_area[0])

    clip_license = src.copy()
    cv2.rectangle(clip_license, (x0, y0), (x1, y1), (0, 0, 255), 10)

    license_img = src[y0:y1,x0:x1]

    gray_license_img = cv2.cvtColor(license_img, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel, iterations = 1)

    # 膨胀使车牌每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    dilate_lic_img = cv2.dilate(binary_license_img, kernel)

    # 轮廓检测，将车牌逐字符分割
    word_images = split_license(dilate_lic_img, binary_license_img)

    # 模板匹配
    result = template_matching(word_images)
    print("".join(result))

    if verbose:

        for img in [gray_license_img, binary_license_img, dilate_lic_img]:
            plt.cla()
            plt.imshow(img, cmap='gray')
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

    task_func = {"1": task2_1, "2": task2_2, "3": task3_3}

    task_func[task_id](verbose=verbose)
