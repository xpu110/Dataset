# import os
# import random
# import shutil
#
#
# # source_file:源路径, target_ir:目标路径
# def cover_files(source_dir, target_ir):
#     for file in os.listdir(source_dir):
#         source_file = os.path.join(source_dir, file)
#
#         if os.path.isfile(source_file):
#             shutil.copy(source_file, target_ir)
#
#
# def ensure_dir_exists(dir_name):
#     """Makes sure the folder exists on disk.
#   Args:
#     dir_name: Path string to the folder we want to create.
#   """
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#
#
# def moveFile(file_dir, save_dir):
#     ensure_dir_exists(save_dir)
#     path_dir = os.listdir(file_dir)  # 取图片的原始路径
#     filenumber = len(path_dir)
#     rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#     picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(path_dir, picknumber)  # 随机选取picknumber数量的样本图片
#     # print (sample)
#     for name in sample:
#         shutil.move(file_dir+'/'+name, save_dir+'/'+name)
#
#
# if __name__ == '__main__':
#     file_dir = '/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/detesets/选好的images'  # 源图片文件夹路径
#     save_dir = '/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/detesets/val'  # 移动到新的文件夹路径
#     moveFile(file_dir,save_dir)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在三个文件夹下
import os
import random
import shutil
from shutil import copy2


#要划分的路径
datadir_normal = r"E:\CODE\image_json"

all_data = os.listdir(datadir_normal)  # （图片文件夹）
all_data_img = []
for i in all_data:
    if i.endswith(".jpg"):
        all_data_img.append(i)
num_all_data = len(all_data_img)
print("num_all_data: " + str(num_all_data))
index_list = list(range(num_all_data))
print(index_list)
random.shuffle(index_list)
num = 0

trainDir = r"E:\CODE\image_json_train"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainDir):
    os.mkdir(trainDir)

validDir = r"E:\CODE\image_json_val"  # （将验证集放在这个文件夹下）
if not os.path.exists(validDir):
    os.mkdir(validDir)

# testDir = './test/'  # （将测试集放在这个文件夹下）
# if not os.path.exists(testDir):
#     os.mkdir(testDir)

for i in index_list:
    fileName = os.path.join(datadir_normal, all_data_img[i])
    json_file = all_data_img[i].split('.')[0] + ".json"
    json_fileName = os.path.join(datadir_normal,json_file)
    if num < num_all_data * 0.8:
        copy2(fileName, trainDir)
        copy2(json_fileName,trainDir)
    else:# num > num_all_data * 0.8 and num < num_all_data * 1:
        # print(str(fileName))
        copy2(fileName, validDir)
        copy2(json_fileName,validDir)
    # else:
    #     copy2(fileName, testDir)
    #     copy2(json_fileName,testDir)
    num += 1
