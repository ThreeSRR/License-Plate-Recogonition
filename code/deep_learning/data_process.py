import cv2
import os
import argparse
import random
from imutils import paths
import numpy as np
from tqdm import trange


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", \
            "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", \
            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', \
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def main(image_dir, train_dir, val_dir):

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)

    img_paths = []
    img_paths += [el for el in paths.list_images(image_dir)]
    random.shuffle(img_paths)

    idx = 0
    idx_train = 0
    idx_val = 0
    for i in trange(len(img_paths)):
        filename = img_paths[i]
        
        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname_split = imgname.split('-')
        rec_x1y1 = imgname_split[2].split('_')[0].split('&')
        rec_x2y2 = imgname_split[2].split('_')[1].split('&')  
        x1, y1, x2, y2 = int(rec_x1y1[0]), int(rec_x1y1[1]), int(rec_x2y2[0]), int(rec_x2y2[1])
        w = int(x2 - x1 + 1.0)
        h = int(y2 - y1 + 1.0)
        
        img = cv2.imread(filename)
        img_crop = np.zeros((h, w, 3))
        img_crop = img[y1:y2+1, x1:x2+1, :]
        
        pre_label = imgname_split[4].split('_')
        lb = ""
        lb += provinces[int(pre_label[0])]
        lb += alphabets[int(pre_label[1])]
        for label in pre_label[2:]:
            lb += ads[int(label)]
            
        idx += 1 
        
        if idx % 4 == 0:
            save = val_dir+'/'+lb+suffix      
            cv2.imwrite(save, img_crop)
            idx_val += 1
        else:
            save = train_dir+'/'+lb+suffix      
            cv2.imwrite(save, img_crop)
            idx_train += 1
            
    print('training images: %d, val images: %d, total: %d' % (idx_train, idx_val, idx))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='crop the licence plate from original image')
    parser.add_argument("--image_dir", default='/data/datasets/CCPD2019/ccpd_base', type=str)
    parser.add_argument("--train_dir", default='./data/train', type=str)
    parser.add_argument("--val_dir", default='./data/validation', type=str)

    args = parser.parse_args()
    image_dir = args.image_dir
    train_dir = args.train_dir
    val_dir = args.val_dir

    main(image_dir, train_dir, val_dir)
