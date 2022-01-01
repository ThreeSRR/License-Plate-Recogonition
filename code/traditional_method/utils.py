import cv2
import numpy as np
from glob import glob
from PIL import ImageFont, ImageDraw, Image


template_path = './template_data/'

num_chars = ['0','1','2','3','4','5','6','7','8','9']
eng_chars = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
chn_chars = ['藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁', \
             '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
num_eng_chars = num_chars + eng_chars

chn_words_list = [glob(template_path+chn_char+'/*') for chn_char in chn_chars]
eng_words_list = [glob(template_path+eng_char+'/*') for eng_char in eng_chars]
eng_num_words_list = [glob(template_path+char+'/*') for char in num_eng_chars]

mapping = {'chn': (chn_chars, chn_words_list), \
           'eng': (eng_chars, eng_words_list), \
           'num': (num_eng_chars, eng_num_words_list)}


def template_score(template, image):
    
    template_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    
    height, width = image.shape
    template_img = cv2.resize(template_img, (width, height))
    result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
    
    return result[0][0]


def template_matching(word_images):
    results = []
    for index, word_image in enumerate(word_images):
        if index == 0:
            chars, words_list = mapping['chn'][0], mapping['chn'][1]
        elif index == 1:
            chars, words_list = mapping['eng'][0], mapping['eng'][1]
        else:
            chars, words_list = mapping['num'][0], mapping['num'][1]
        
        best_score = []
        for word_list in words_list:
            score = []
            for word in word_list:
                res = template_score(word, word_image)
                score.append(res)
            best_score.append(max(score))
        idx = best_score.index(max(best_score))
        r = chars[idx]
        results.append(r)

    return results


def split_license(src, dst):

    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        words.append(rect)

    words = sorted(words,key=lambda s:s[0],reverse=False)
    for word in words:
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
            split_image = dst[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(split_image)
            
    return word_images


def cv2ImgAddText(img, text, pos, textColor=(200, 0, 0), textSize=200):
    
    fontpath = "../../resources/simsun.ttc"
    font = ImageFont.truetype(fontpath, textSize, encoding='utf-8')
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, textColor, font=font)

    return img
