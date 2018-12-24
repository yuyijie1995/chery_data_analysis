# -*- coding=utf-8 -*-
import json
import os
import cv2
import numpy as np
import skimage.io
import xml.etree.ElementTree as ET
import shutil
import PIL.Image
import PIL.ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils




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
    binary_mask_encoded = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = maskUtils.area(binary_mask_encoded)
    return mask,area



all_label_names=['car','bus','engineeringcar','rickshaw','line_2','minibus','bigtruck','people','smalltruck',
'line_1',
'line_9',
'line_0',
'cyclist',
'line_13',
'line_7',
'person',
'line_6']
# 从json文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def position(pos):#该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x=[]
    y=[]
    nums=len(pos)-1
    for i in range(0,nums,2):
        x.append(pos[i])
        y.append(pos[i+1])
    x_max=max(x)
    x_min=min(x)
    y_max=max(y)
    y_min=min(y)
    b=(int(x_min),int(y_min),int(x_max),int(y_max))
    return b
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



def parse_json(json_path):
    with open(json_path,'r') as LabelFile:
        load_dict=json.load(LabelFile)
        coords=list()
        polygons=list()
        for i,object_dict in enumerate(load_dict['shapes']):
            polygon=[]

            if object_dict['label']  in all_label_names and object_dict['shape_type']=='rectangle':
                name=object_dict['label']
                if name in ['car','bus','engineeringcar','minibus','bigtruck','smalltruck']:
                    name='car'
                if name in ['rickshaw','cyclist']:
                    name='cyclist'
                if name in ['people','person']:
                    name='person'
                if name in ['line_2','line_13']:
                    name='line_13'
                bb=position2(object_dict['points'])
                x_min=int(bb[0])
                y_min=int(bb[2])
                x_max = int(bb[1])
                y_max = int(bb[3])
                coords.append([x_min,y_min,x_max,y_max,name])
            elif object_dict['label']  in all_label_names and object_dict['shape_type']=='polygon':
                name = object_dict['label']
                if name in ['car', 'bus', 'engineeringcar', 'minibus', 'bigtruck', 'smalltruck']:
                    name = 'car'
                if name in ['rickshaw', 'cyclist']:
                    name = 'cyclist'
                if name in ['people', 'person']:
                    name = 'person'
                if name in ['line_2', 'line_13']:
                    name = 'line_13'
                polygons_points=object_dict['points']
                nums = len(polygons_points)
                for i in range(nums):
                    polygon.append(polygons_points[i][0])
                    polygon.append(polygons_points[i][1])
                polygon.append(name)
                polygons.append(polygon)
    return coords,polygons


    # tree = ET.parse(xml_path)
    # root = tree.getroot()
    # objs = root.findall('object')
    # coords = list()
    # for ix, obj in enumerate(objs):
    #     name = obj.find('name').text
    #     box = obj.find('bndbox')
    #     x_min = int(box[0].text)
    #     y_min = int(box[1].text)
    #     x_max = int(box[2].text)
    #     y_max = int(box[3].text)
    #     coords.append([x_min, y_min, x_max, y_max, name])
    # return coords

def convert(root_path, source_xml_root_path, target_xml_root_path, phase='train', split=80000):
    '''
    root_path:
        根路径，里面包含JPEGImages(图片文件夹)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
    source_xml_root_path:
        VOC xml文件存放的根目录
    target_xml_root_path:
        coco xml存放的根目录
    phase:
        状态：'train'或者'test'
    split:
        train和test图片的分界点数目

    '''

    dataset = {'categories':[], 'images':[], 'annotations':[]}

    # 打开类别标签
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()

    # 建立类别标签和数字id的对应关系
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'beverage'})   #mark

    # 读取images文件夹的图片名称
    pics = [f for f in os.listdir(os.path.join(root_path, 'images'))]

    # 判断是建立训练集还是验证集
    if phase == 'train':
        pics = [line for i, line in enumerate(pics) if i <= split]
    elif phase == 'val':
        pics = [line for i, line in enumerate(pics) if i > split]

    print('---------------- start convert ---------------')
    bnd_id = 1	#初始为1
    for i, pic in enumerate(pics):
        # print('pic  '+str(i+1)+'/'+str(len(pics)))
        xml_path = os.path.join(source_xml_root_path, pic[:-4]+'.json')
        pic_path = os.path.join(root_path, 'images/' + pic)
        # xml_path = os.path.join(source_xml_root_path, '20180726T145313T4256' + '.json')
        # pic_path = os.path.join(root_path, 'images/' + '20180726T145313T4256.jpg')
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(pic_path)
        height, width, _ = im.shape
        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': pic,
                                  'id': i,
                                  'width': width,
                                  'height': height})
        try:
            coords,polygons = parse_json(xml_path)
        except:
            print(pic[:-4]+'.xml not exists~')
            continue
        for coord in coords:
            # x_min
            x1 = int(coord[0])-1
            x1 = max(x1, 0)
            # y_min
            y1 = int(coord[1])-1
            y1 = max(y1, 0)
            # x_max
            x2 = int(coord[2])
            # y_max
            y2 = int(coord[3])
            assert x1<x2
            assert y1<y2
            # name
            name = coord[4]
            cls_id = classes.index(name)+1	#从1开始
            width_box = max(0, x2 - x1)
            height_box = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width_box * height_box,
                'bbox': [x1, y1, width_box, height_box],
                'category_id': int(cls_id),
                'id': bnd_id,
                'image_id': i,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            bnd_id += 1
        for polygon in polygons:
            # x_min
            polygon_origin=[]
            for i in range(0,len(polygon)-1,2):
                polygon_origin.append([polygon[i],polygon[i+1]])

            polygon_mask,area=polygons_to_mask(im.shape[:2],polygon_origin)
            # name
            name = polygon[-1]
            cls_id = classes.index(name)+1	#从1开始
            bbox=position(polygon)
            xp1 = int(bbox[0]) - 1
            xp1 = max(xp1, 0)
            # y_min
            yp1 = int(bbox[1]) - 1
            yp1 = max(yp1, 0)
            # x_max
            xp2 = int(bbox[2])
            # y_max
            yp2 = int(bbox[3])
            widthp = max(0, xp2 - xp1)
            heightp = max(0, yp2 - yp1)

            #area2=widthp*heightp
            #array_polygon = np.array(polygon)
            #area=cv2.contourArea(array_polygon)
            dataset['annotations'].append({
                'area': int(area),
                'bbox': [xp1, yp1, widthp, heightp],
                'category_id': int(cls_id),
                'id': bnd_id,
                'image_id': i,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[point for point in polygon[:-1]]]
            })
            bnd_id += 1


    # 保存结果的文件夹
    folder = os.path.join(target_xml_root_path, 'annotations')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    json_name = os.path.join(target_xml_root_path, 'annotations/instances_{}2014.json'.format(phase))
    with open(json_name, 'w') as f:
      json.dump(dataset, f)

if __name__ == '__main__':
    convert(root_path='/media/wrc/8EF06A4CF06A3A9B/qirui/dataset', source_xml_root_path = '/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/labels', target_xml_root_path = './data_coco')