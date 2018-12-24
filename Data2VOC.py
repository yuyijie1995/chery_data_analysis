#! /usr/bin/python
# -*- coding:UTF-8 -*-
import os, sys
import glob
from PIL import Image
import json

# VEDAI 图像存储位置
src_img_dir = "/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/images"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
json_path = "/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/labels"
src_xml_dir = "/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/Annotations"
all_label_names=['car','bus','engineeringcar','rickshaw','line_2','minibus','bigtruck','people','smalltruck',
'line_1',
'line_9',
'line_0',
'cyclist',
'line_13',
'line_7',
'person',
'line_6']
img_names = os.listdir('/media/wrc/8EF06A4CF06A3A9B/qirui/dataset/labels')

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
for img in img_names:
    img = img[:-5]
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    # open the crospronding txt file
    json_path1=os.path.join(json_path,img+'.json')
    with open(json_path1,'r') as LabelFile:
        load_dict=json.load(LabelFile)
        coords=list()
        for i,object_dict in enumerate(load_dict['shapes']):
            if object_dict['label']  in all_label_names and object_dict['shape_type']=='rectangle':
                name=object_dict['label']
                if name in ['car','bus','engineeringcar','minibus','bigtruck','smalltruck']:
                    name='car'
                if name in ['rickshaw','cyclist']:
                    name='cyclist'
                if name in ['people','person']:
                    name='person'
                bb=position2(object_dict['points'])
                x_min=int(bb[0])
                y_min=int(bb[2])
                x_max = int(bb[1])
                y_max = int(bb[3])
                coords.append([x_min,y_min,x_max,y_max,name])

    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')


    # write the region of image on xml file
        for img_each_label in coords:
            spt = img_each_label  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            # spt[0] = 'helmet'
            xml_file.write('        <name>' + str(spt[-1]) + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(spt[0])) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(spt[1])) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(spt[2])) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(spt[3])) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        # xml_file.write('        <name>' + str(spt[0]) + '</name>\n')
        # xml_file.write('        <pose>Unspecified</pose>\n')
        # xml_file.write('        <truncated>0</truncated>\n')
        # xml_file.write('        <difficult>0</difficult>\n')
        # xml_file.write('        <bndbox>\n')
        # xml_file.write('            <xmin>' + str(int(spt[4])) + '</xmin>\n')
        # xml_file.write('            <ymin>' + str(int(spt[5])) + '</ymin>\n')
        # xml_file.write('            <xmax>' + str(int(spt[6])) + '</xmax>\n')
        # xml_file.write('            <ymax>' + str(int(spt[7])) + '</ymax>\n')
        # xml_file.write('        </bndbox>\n')
        # xml_file.write('    </object>\n')

    xml_file.write('</annotation>')