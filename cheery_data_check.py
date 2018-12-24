import cv2
from os import listdir, getcwd
from os.path import join
import os.path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import csv
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import pandas as pd
import glob
import shutil
from skimage.data import imread

data_dir='D:/postgraduateworking/dataset/qirui/dataset'
out_image_dir='D:/postgraduateworking/dataset/qirui/dataset/images'
label_dir='D:/postgraduateworking/dataset/qirui/dataset/labels'
if not os.path.exists(out_image_dir):
    os.mkdir(out_image_dir)
files=glob.glob(data_dir+'/*.jpg')
imagefiles=glob.glob(out_image_dir+'/*.jpg')
all_label_name={}
strage_class=['rickshaw','other','incomplete','person','people','ca','uncomplete','obstacle']
strage_class_check='../strage'
if not os.path.exists(strage_class_check):
    os.mkdir(strage_class_check)
count=0

def position(pos):#该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
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
    b=(int(x_min),int(x_max),int(y_min),int(y_max))
    return b

for file in imagefiles:
    #shutil.move(file,out_image_dir)
    _,FileName=os.path.split(file)
    FileName2,_=os.path.splitext(FileName)
    LabelFilePath=os.path.join(label_dir,'%s.json'%FileName2)
    with open(LabelFilePath,'r') as LabelFile:
        load_dict=json.load(LabelFile)
        for i,object_dict in enumerate(load_dict['shapes']):

            if object_dict['label'] not in all_label_name :
                labelnamedir='../%s'%object_dict['label']
                if not os.path.exists(labelnamedir):
                    os.mkdir(labelnamedir)
                all_label_name['%s'%object_dict['label']]=1
                # img=cv2.imread(file)
                # if object_dict['shape_type']=='rectangle':
                #     b = position(object_dict['points'])
                #     cropimage = img[b[2]:b[3], b[0]:b[1]]
                #
                #     cv2.imwrite('D:/postgraduateworking/kaggle/kaggle_airbus/%s/%s%s.png'%(object_dict['label'],FileName2,i),cropimage)
                # elif object_dict['shape_type']=='polygon':
                #     b=position(object_dict['points'])
                #     cropimage=img[b[2]:b[3],b[0]:b[1]]
                #     cv2.imwrite('D:/postgraduateworking/kaggle/kaggle_airbus/%s/%s%s.png'%(object_dict['label'],FileName2,i),cropimage)

            else:
                all_label_name['%s'%object_dict['label']]+=1
                # labelnamedir = '../%s' % object_dict['label']
                # if not os.path.exists(labelnamedir):
                #     os.mkdir(labelnamedir)
                # all_label_name['%s' % object_dict['label']] = 1
                # img = cv2.imread(file)
                # if object_dict['shape_type'] == 'rectangle':
                #     b = position(object_dict['points'])
                #     cropimage = img[b[2]:b[3], b[0]:b[1]]
                #
                #     cv2.imwrite('D:/postgraduateworking/kaggle/kaggle_airbus/%s/%s%s.png' % (
                #     object_dict['label'], FileName2,i), cropimage)
                # elif object_dict['shape_type'] == 'polygon':
                #     b = position(object_dict['points'])
                #     cropimage = img[b[2]:b[3], b[0]:b[1]]
                #     cv2.imwrite('D:/postgraduateworking/kaggle/kaggle_airbus/%s/%s%s.png' % (
                #     object_dict['label'], FileName2,i), cropimage)
            # if object_dict['label']  in strage_class :
            #     shutil.copy(LabelFilePath,strage_class_check)
            #     shutil.copy(file,strage_class_check)
            # else:
            #     all_label_name['%s'%object_dict['label']]+=1
print(all_label_name)
# with open('../classname.csv', 'w') as f:
#     w = csv.writer(f)
#     # write each key/value pair on a separate row
#     w.writerows(all_label_name.items())

X=[classname for classname in all_label_name.keys()]
Y=[classnumber for classnumber in all_label_name.values()]
print(X,Y)

fig = plt.figure(figsize=(25,25))
plt.bar(X, Y, 0.4, color="green")
plt.xticks(rotation=330)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("ClassNumberCheck")


plt.savefig("../ClassNumberCheck.jpg")
plt.show()






