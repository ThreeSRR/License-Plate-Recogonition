import numpy as np
import argparse
import torch
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt 

from models import LPRNet, STNet
from utils import cv2ImgAddText, decode, CHARS


def test(lprnet_path, stn_path, image_path):

    dir_path = './deep_learning_res'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load(lprnet_path, map_location=lambda storage, loc: storage))
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load(stn_path, map_location=lambda storage, loc: storage))
    
    lprnet.eval()
    STN.eval()
    
    image = cv2.imread(image_path)
    img = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    img = (np.transpose(np.float32(img), (2, 0, 1)) - 127.5)*0.0078125
    data = torch.from_numpy(img).float().unsqueeze(0).to(device)
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()
    
    labels, _ = decode(preds, CHARS)

    print(labels[0])
            
    img = cv2ImgAddText(image, labels[0], (0, 0))

    img_name = os.path.split(image_path)[-1]
    plt.cla()
    
    plt.imsave(os.path.join(dir_path, img_name), img[:,:,::-1])
    plt.imshow(img[:,:,::-1])
    plt.show()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("--image", help='image path', default='./plate_images/3-3.jpg', type=str)
    parser.add_argument("--lprnet_path", help='lprnet path', default='./weights/best_LPRNet_model.pth', type=str)
    parser.add_argument("--stn_path", help='stn path', default='./weights/best_STN_model.pth', type=str)

    args = parser.parse_args()
    lprnet_path = args.lprnet_path
    stn_path = args.stn_path
    image_path = args.image

    for image_path in glob('./plate_images/*.jpg'):
        test(lprnet_path, stn_path, image_path)
