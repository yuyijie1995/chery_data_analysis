import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import pandas as pd
import PIL.Image
import PIL.ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from itertools import groupby
from skimage import measure
from skimage.data import imread
import matplotlib.pyplot as plt
import cv2

dataset_train = '/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/images'
label_train = '/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/labels'
IMAGE_DIR = dataset_train

#df = pd.read_csv(csv_train)  # read csv file

INFO = {
    "description": "Qirui Dataset",
    "url": "https://github.com/yuyijie1995",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "yuyijie1995",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': 'beverage',
    },
{
        'id': 2,
        'name': 'cyclist',
        'supercategory': 'beverage',
    },
{
        'id': 3,
        'name': 'person',
        'supercategory': 'beverage',
    },
{
        'id': 4,
        'name': 'line',
        'supercategory': 'beverage',
    },
]
classes={'car':1,'cyclist':2,'person':3,'line':4}
all_label_names=['car','bus','engineeringcar','rickshaw','line_2','minibus','bigtruck','people','smalltruck',
'line_1',
'line_9',
'line_0',
'cyclist',
'line_13',
'line_7',
'person',
'line_6']
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons



def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def save_bad_ann(image_name, mask, segmentation_id):
    img = imread(os.path.join(IMAGE_DIR, image_name))
    fig, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    plt.savefig(os.path.join('./tmp', image_name.split('.')[0] + '_' + str(segmentation_id) + '.png'))
    plt.close()

def position2(pos):#该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x=[]
    y=[]
    nums=len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max=max(x)
    x_min=min(x)
    y_max=max(y)
    y_min=min(y)
    b=(float(x_min),float(x_max),float(y_min),float(y_max))
    return b

def polygons_to_mask(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    mask2=mask.astype(int)
    binary_mask_encoded = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = maskUtils.area(binary_mask_encoded)
    box=maskUtils.toBbox(binary_mask_encoded)
    segmentation = binary_mask_to_polygon(mask2, 2)
    return mask,area,segmentation,mask2,box

def parse_json(json_path):
    with open(json_path,'r') as LabelFile:
        load_dict=json.load(LabelFile)
        polygons=list()
        for i,object_dict in enumerate(load_dict['shapes']):
            polygon=[]


            if object_dict['label']  in all_label_names :
                name = object_dict['label']
                if name in ['car', 'bus', 'engineeringcar', 'minibus', 'bigtruck', 'smalltruck']:
                    name = 'car'
                if name in ['rickshaw', 'cyclist']:
                    name = 'cyclist'
                if name in ['people', 'person']:
                    name = 'person'
                if name in ['line_2', 'line_13','line_1','line_9','line_0','line_7','line_6']:
                    name = 'line'
                polygons_points=object_dict['points']
                nums = len(polygons_points)
                for i in range(nums):
                    polygon.append(polygons_points[i][0])
                    polygon.append(polygons_points[i][1])
                polygon.append(name)
                polygons.append(polygon)
    return polygons

def main():
    # 最终放进json文件里的字典
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],  # 放一个空列表占位置，后面再append
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # 最外层的循环是图片，因为图片的基本信息需要共享
    # IMAGE_DIR路径下找到所有的图片
    for root, _, files in os.walk(IMAGE_DIR):
        image_paths = filter_for_jpeg(root, files)  # 图片文件地址
        num_of_image_files = len(image_paths)  # 图片个数

        # 遍历每一张图片
        for image_path in image_paths:
            # 提取图片信息
            image = Image.open(image_path)
            im = cv2.imread(image_path)
            image_name = os.path.basename(image_path)  # 不需要具体的路径，只要图片文件名
            image_info = pycococreatortools.create_image_info(
                image_id, image_name, image.size)
            coco_output["images"].append(image_info)
            json_path=os.path.join(label_train,image_name[:-4]+'.json')
            with open(json_path, 'r') as LabelFile:
                load_dict = json.load(LabelFile)
            # 内层循环是mask，把每一张图片的mask搜索出来
                targets=load_dict['shapes']
                num_of_targets=len(targets)
            polygons=parse_json(json_path)
            #rle_masks = df.loc[df['ImageId'] == image_name, 'EncodedPixels'].tolist()
            #num_of_rle_masks = len(rle_masks)
            for polygon in polygons:
                # x_min
                polygon_origin = []
                for i in range(0, len(polygon) - 1, 2):
                    polygon_origin.append([polygon[i], polygon[i + 1]])
                #size_list=list(image.size()[::-1])
                if len(polygon)<6:
                    bbox = position2(polygon_origin)
                    x1 = int(bbox[0]) - 1
                    x1 = max(x1, 0)
                    # y_min
                    y1 = int(bbox[2]) - 1
                    y1 = max(y1, 0)
                    # x_max
                    x2 = int(bbox[1])
                    # y_max
                    y2 = int(bbox[3])

                    width_box = max(0, x2 - x1)
                    height_box = max(0, y2 - y1)
                    area=width_box*height_box
                    bounding_box = [x1, y1, width_box, height_box]
                    class_id = classes['%s' % polygon[-1]]
                    category_info = {'id': class_id, 'is_crowd': 0}
                    annotation_info = {
                        "id": segmentation_id,
                        "image_id": image_id,
                        "category_id": category_info["id"],
                        "iscrowd": 0,
                        "area": area,
                        "bbox": bounding_box,
                        "segmentation": [[x1,y1,x2,y1,x2,y2,x1,y2]],
                        "width": width_box,
                        "height": height_box,
                    }

                    # 不是所有的标注都会被转换,低质量标注会被过滤掉
                    # 正常的标注加入数据集，不好的标注保存供观察
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    # else:
                    #     save_bad_ann(image_name, binary_mask, segmentation_id)

                    # 无论标注是否被写入数据集，均分配一个编号
                    segmentation_id = segmentation_id + 1
                else:
                    encode_mask,area,segmentation,binary_mask,box=polygons_to_mask(im.shape[:2],polygon_origin)
                    # bbox = position2(polygon_origin)
                    # xp1 = int(bbox[0]) - 1
                    # xp1 = max(xp1, 0)
                    # # y_min
                    # yp1 = int(bbox[2]) - 1
                    # yp1 = max(yp1, 0)
                    # # x_max
                    # xp2 = int(bbox[1])
                    # # y_max
                    # yp2 = int(bbox[3])
                    # bounding_box=[xp1,yp1,xp2,yp2]
                #binary_mask = rle_decode(rle_masks[index])
                    class_id = classes['%s'%polygon[-1]]
                    category_info = {'id': class_id, 'is_crowd': 0}
                    annotation_info = {
                        "id": segmentation_id,
                        "image_id": image_id,
                        "category_id": category_info["id"],
                        "iscrowd": 0,
                        "area": area.tolist(),
                        "bbox": box.tolist(),
                        "segmentation": segmentation,
                        "width": binary_mask.shape[1],
                        "height": binary_mask.shape[0],
                    }

                # 不是所有的标注都会被转换,低质量标注会被过滤掉
                # 正常的标注加入数据集，不好的标注保存供观察
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                else:
                    save_bad_ann(image_name, binary_mask, segmentation_id)

                # 无论标注是否被写入数据集，均分配一个编号
                segmentation_id = segmentation_id + 1

            print("%d of %d is done." % (image_id, num_of_image_files))
            image_id = image_id + 1

    with open('../instances_qirui_train2018.json', 'w') as output_json_file:
        # json.dump(coco_output, output_json_file)
        json.dump(coco_output, output_json_file, indent=4)


if __name__ == "__main__":
    main()
