#encoding=utf-8
#数据集扩充程序，只能扩充json格式的数据
import base64
import json
import shutil

from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re
import os
import os.path as osp
from tqdm import tqdm

class DataAugment(object):
    def __init__(self, image_id='', save_dir=''):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.id = image_id
        path = str(self.id)+'.jpg'
        img = cv.imread(path,-1)
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+'.jpg')
            sys.exit(0)
        self.src = img                                           #原图
        #dst1 = cv.flip(img, 0, dst=None)                        #上下反转
        dst2 = cv.flip(img, 1, dst=None)                 #左右反转
        #dst3 = cv.flip(img, -1, dst=None)                       #上下左右反转
        #self.flip_x = dst1
        self.flip_y = dst2
        #self.flip_x_y = dst3
        #cv.imwrite(save_dir + '/' + img_id + '_flip_x' + '.jpg', self.flip_x)
        #cv.imwrite(save_dir + '/' + img_id + '.jpg', self.src)
        cv.imwrite(save_dir + '/' + img_id + '_flip_y'+'.jpg', self.flip_y)
        #cv.imwrite(save_dir + '/' + img_id + '_flip_x_y' + '.jpg', self.flip_x_y)

    #添加高斯噪声
    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            #dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            #dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(save_dir + '/' + img_id + '_Gaussian' + '.jpg', dst1)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x' + '_Gaussian' + '.jpg', dst2)
            cv.imwrite(save_dir + '/' + img_id + '_flip_y' + '_Gaussian'+'.jpg', dst3)
            #cv.imwrite(save_dir + '/' + img_id+ '_flip_x_y' + '_Gaussian' + '.jpg', dst4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.9
            increase = 1.1
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            #dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
            #dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
            dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
            dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
            #dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            #dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            cv.imwrite(save_dir + '/' + img_id + '_ReduceEp' + '.jpg', dst1)
            #cv.imwrite(save_dir + '/' + img_id +'_flip_x' + '_ReduceEp' + '.jpg', dst3)
            cv.imwrite(save_dir + '/' + img_id + '_flip_y' + '_ReduceEp' + '.jpg', dst5)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x_y' + '_ReduceEp' + '.jpg', dst7)
            cv.imwrite(save_dir + '/' + img_id + '_IncreaseEp' + '.jpg', dst2)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x' + '_IncreaseEp' + '.jpg', dst4)
            cv.imwrite(save_dir + '/' + img_id + '_flip_y' + '_IncreaseEp' + '.jpg', dst6)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x_y' + '_IncreaseEp' + '.jpg', dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.03
            dst1 = self.src
            #dst2 = self.flip_x
            dst3 = self.flip_y
            #dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    #dst2[rand_x, rand_y] = 0
                    dst3[rand_x, rand_y] = 0
                    #dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    #dst2[rand_x, rand_y] = 255
                    dst3[rand_x, rand_y] = 255
                    #dst4[rand_x, rand_y] = 255
            cv.imwrite(save_dir + '/' + img_id + '_Salt' + '.jpg', dst1)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x' + '_Salt' + '.jpg', dst2)
            cv.imwrite(save_dir + '/' + img_id + '_flip_y' + '_Salt' + '.jpg', dst3)
            #cv.imwrite(save_dir + '/' + img_id + '_flip_x_y' + '_Salt' + '.jpg', dst4)

    def json_generation(self):
        image_names = [save_dir + '/' + img_id + '_flip_y']#,save_dir+img_id+'_flip_x_y',save_dir+img_id+'_flip_x']
        if self.gaussianBlur:
            image_names.append(save_dir + '/' + img_id + '_Gaussian')
            #image_names.append(save_dir+ '/' + img_id + '_flip_x' + '_Gaussian')
            image_names.append(save_dir + '/' + img_id + '_flip_y' + '_Gaussian')
            #image_names.append(save_dir + '/' + img_id + '_flip_x_y' + '_Gaussian')
        if self.changeExposure:
            image_names.append(save_dir + '/' + img_id + '_ReduceEp')
            #image_names.append(save_dir + '/' + img_id + '_flip_x' + '_ReduceEp')
            image_names.append(save_dir + '/' + img_id + '_flip_y' + '_ReduceEp')
            #image_names.append(save_dir + '/' + img_id + '_flip_x_y' + '_ReduceEp')
            image_names.append(save_dir + '/' + img_id + '_IncreaseEp')
            #image_names.append(save_dir + '/' + img_id + '_flip_x' + '_IncreaseEp')
            image_names.append(save_dir + '/' + img_id + '_flip_y'+'_IncreaseEp')
            #image_names.append(save_dir + '/' + img_id +'_flip_x_y' + '_IncreaseEp')
        if self.add_saltNoise:
            image_names.append(save_dir + '/' + img_id + '_Salt')
            #image_names.append(save_dir + '/' + img_id + '_flip_x' + '_Salt')
            image_names.append(save_dir + '/' + img_id + '_flip_y' + '_Salt')
            #image_names.append(save_dir + '/' + img_id + '_flip_x_y' + '_Salt')
        for image_name in image_names:
            with open(image_name + ".jpg", "rb")as b64:
                base64_data_original = str(base64.b64encode(b64.read()))
                # In pycharm:
                match_pattern=re.compile(r'b\'(.*)\'')
                base64_data=match_pattern.match(base64_data_original).group(1)
                # In terminal:
                #base64_data = base64_data_original
            with open(str(self.id)+".json", 'r',encoding="utf-8")as js:
                json_data = json.load(js)
                img = utils.img_b64_to_arr(json_data['imageData'])
                height, width = img.shape[:2]
                shapes = json_data['shapes']
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        match_pattern2 = re.compile(r'(.*)_x(.*)')
                        match_pattern3 = re.compile(r'(.*)_y(.*)')
                        match_pattern4 = re.compile(r'(.*)_x_y(.*)')
                        if match_pattern4.match(image_name):
                            point[0] = width - point[0]
                            point[1] = height - point[1]
                        elif match_pattern3.match(image_name):
                            point[0] = width - point[0]
                            point[1] = point[1]
                        elif match_pattern2.match(image_name):
                            point[0] = point[0]
                            point[1] = height - point[1]
                        else:
                            point[0] = point[0]
                            point[1] = point[1]
                    for point in points:
                        if points[0][0] > points[1][0]:
                            points[0][0], points[1][0] = points[1][0], points[0][0]
                json_data['imagePath'] = image_name.split('/')[-1]+".jpg"
                json_data['imageData'] = base64_data
                json.dump(json_data, open(image_name+".json", 'w'), indent=2)

#将原图片与标签复制到扩充文件夹
def copy_file(file_dir, save_dir):
    filelist = os.listdir(file_dir)      #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(file_dir, file)
        dst = os.path.join(save_dir, file)
        shutil.copy(src, dst)


if __name__ == "__main__":
    file_dir = r'E:\CODE\image_json_val'    #图片和json标签的路径
    save_dir = r'E:\CODE\image_json_val_enhance'    #扩充之后的输出路径
    if not osp.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    for img_name in tqdm(os.listdir(file_dir)):
        img_id = os.path.splitext(img_name)[0]
        dataAugmentObject = DataAugment(file_dir + '/' + img_id, save_dir)
        dataAugmentObject.gaussian_blur_fun()
        dataAugmentObject.change_exposure_fun()
        dataAugmentObject.add_salt_noise()
        dataAugmentObject.json_generation()
    copy_file(file_dir, save_dir)