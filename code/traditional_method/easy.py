import cv2
import argparse
import matplotlib.pyplot as plt
from utils import template_matching, split_license, cv2ImgAddText


def task1_1(verbose=True):
    src = cv2.imread('../../resources/easy/1-1.jpg')

    gray_license_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 膨胀，使每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 20))
    dilate_lic_img = cv2.dilate(binary_license_img, kernel)

    # 轮廓检测将车牌逐字符分割
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


def task1_2(verbose=True):
    src = cv2.imread('../../resources/easy/1-2.jpg')

    gray_license_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)

    binary_license_img[binary_license_img==0] = 1
    binary_license_img[binary_license_img==255] = 0
    binary_license_img[binary_license_img==1] = 255

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 25))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 膨胀，使每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    dilate_lic_img = cv2.dilate(binary_license_img, kernel)

    # 轮廓检测将车牌逐字符分割
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


def task1_3(verbose=True):
    src = cv2.imread('../../resources/easy/1-3.jpg')

    gray_license_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将车牌二值化
    _, binary_license_img = cv2.threshold(gray_license_img, 0, 255, cv2.THRESH_OTSU)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binary_license_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_OPEN, kernel,iterations = 1)

    # 闭运算，使每个字符连成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 40))
    dilate_lic_img = cv2.morphologyEx(binary_license_img, cv2.MORPH_CLOSE, kernel,iterations = 1)

    # 轮廓检测将车牌逐字符分割
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

    task_func = {"1": task1_1, "2": task1_2, "3": task1_3}

    task_func[task_id](verbose=verbose)
