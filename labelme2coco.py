# 命令行执行： python labelme2coco.py --i [图片和标签的文件夹] --o [输出文件夹] --l [labels.txt，类名的文件]
# 输出文件夹必须为空文件夹

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import imgviz
import numpy as np
import labelme
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split   #pip install scikit-learn
from PIL import Image

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")  #https://pypi.tuna.tsinghua.edu.cn/simple/pycocotools-windows/
    sys.exit(1)


def to_coco(args, label_files, train):
    # 创建 总标签data

    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    class_name_to_id = {}
    for i, line in enumerate(open(args.l).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()  # strip() 方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列。

        if class_id == -1:
            assert class_name == "__ignore__"  # class1:0, class2:1, .......
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name, )
        )

    if train:
        out_ann_file = osp.join(args.o, "annotations", "instances_train2017.json")
    else:
        out_ann_file = osp.join(args.o, "annotations", "instances_val2017.json")



    for image_id, filename in enumerate(label_files):
        print("filename=",filename)
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]  # 文件名不带后缀
        if train:
            out_img_file = osp.join(args.o, "train2017", base + ".jpg")
        else:
            out_img_file = osp.join(args.o, "val2017", base + ".jpg")

        print("| ", out_img_file)

        # ************************** 对图片的处理开始 *******************************************
        # 将标签文件对应的图片进行保存到对应的 文件夹。train保存到 train2017/ test保存到 val2017/
        img = labelme.utils.img_data_to_arr(label_file.imageData)  # .json文件中包含图像，用函数提出来
        imgviz.io.imsave(out_img_file, img)  # 将图像保存到输出路径

        # ************************** 对图片的处理结束 *******************************************

        # ************************** 对标签的处理开始 *******************************************
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name= base + ".jpg",
                #   out_img_file = "/coco/train2017/1.jpg"
                #   out_ann_file = "/coco/annotations/annotations_train2017.json"
                #   osp.dirname(out_ann_file) = "/coco/annotations"
                #   file_name = ..\train2017\1.jpg   out_ann_file文件所在目录下 找 out_img_file 的相对路径
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )
        # ************************** 对标签的处理结束 *******************************************

        # ************************** 可视化的处理开始 *******************************************
        if args.viz:
            labels, captions, masks = zip(
                *[
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, gid), msk in masks.items()
                    if cnm in class_name_to_id
                ]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=30,
                line_width=4,
                alpha=0.2,
                colormap=np.array([[255,56,56], [0,24,236], [255,178,29], [72,249,10]],
                                     dtype=np.uint8)     #颜色RGB格式的
                #'FF3838', '0018EC', 'FFB21D', '48F90A'
            )
            out_viz_file = osp.join(
                args.o, "visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)
        # ************************** 可视化的处理结束 *******************************************

    with open(out_ann_file, "w") as f:  # 将每个标签文件汇总成data后，保存总标签data文件
        json.dump(data, f)


# 主程序执行
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i",  default=r'E:\CODE\image_json_val_enhance',help="input annotated directory")
    parser.add_argument("--o",  default=r'E:\CODE\image_json_val_enhance_coco',help="output dataset directory")
    parser.add_argument("--l",  default='labels.txt', help="labels file")#类名文件
    parser.add_argument("--viz",  default=True ,help="if visualization", action="store_true")#是否可视化，True可视化，False不可视化
    args = parser.parse_args()

    if not osp.exists(args.o):
        os.makedirs(args.o,exist_ok=True)
    print("| Creating dataset dir:", args.o)

    if args.viz:
        os.makedirs(osp.join(args.o, "visualization"),exist_ok=True)

    # 创建保存的文件夹
    if not os.path.exists(osp.join(args.o, "annotations")):
        os.makedirs(osp.join(args.o, "annotations"),exist_ok=True)
    if not os.path.exists(osp.join(args.o, "train2017")):
        os.makedirs(osp.join(args.o, "train2017"),exist_ok=True)
    if not os.path.exists(osp.join(args.o, "val2017")):
        os.makedirs(osp.join(args.o, "val2017"),exist_ok=True)

    # 获取目录下所有的.jpg文件列表
    feature_files = glob.glob(osp.join(args.i, "*.jpg"))
    print('| Image number: ', len(feature_files))

    # 获取目录下所有的joson文件列表
    label_files = glob.glob(osp.join(args.i, "*.json"))
    print('| Json number: ', len(label_files))

    # feature_files:待划分的样本特征集合    label_files:待划分的样本标签集合    test_size:测试集所占比例
    # x_train:划分出的训练集特征      x_test:划分出的测试集特征     y_train:划分出的训练集标签    y_test:划分出的测试集标签
    x_train, x_test, y_train, y_test = train_test_split(feature_files, label_files, test_size=0.0000000000000000000000000000001)
    print("| Train number:", len(y_train), '\t Value number:', len(y_test))

    # 把训练集标签转化为COCO的格式，并将标签对应的图片保存到目录 /train2017/
    print("—" * 50)
    print("| Train images:")
    to_coco(args, y_train, train=False) #train=True，表示制作训练集

    # 把测试集标签转化为COCO的格式，并将标签对应的图片保存到目录 /val2017/
    print("—" * 50)
    print("| Value images:")
    to_coco(args, y_test, train=True) #train=False，表示制作验证集


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)
