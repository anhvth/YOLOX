import mmcv
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from avcv.utils import *
# from avcv.visualize import *
from glob import glob
import os
import imagesize



class Cat2Id:
    def __init__(self):
        self.d = {}

    def __call__(self, cat):
        if not cat in self.d:
            self.d[cat] = len(self.d)
        return self.d[cat]

    def tolist(self):
        out = []
        for name, id in self.d.items():
            out.append(dict(name=name, id=id))
        return out

cat2id = Cat2Id()

def line_to_ann(line, image_id, ann_id):
    x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = line.replace('\n', '').split(' ')
    x1, y1, x2, y2, x3, y3, x4, y4 = [float(_) for _ in [x1, y1, x2, y2, x3, y3, x4, y4]]
    cnt = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([-1, 2])
    x, y, w, h = cv2.boundingRect(cnt.astype('int'))
    ann = dict(
        bbox=[x, y, w, h],
        segmentation=[[x1, y1, x2, y2, x3, y3, x4, y4]],
        category_id=cat2id(category),
        id=ann_id,
        iscrowd=0,
        area=h*w,
        image_id=image_id,
    )
    return ann
def f(dset):
    txt_paths = glob(f"/home/av/gitprojects/yolox/datasets/dota/2.0/{dset}/labelTxt-v2.0/*.txt")
    print('Num of sample training:', len(txt_paths))
    IMG_DIR_1 = f"/home/av/gitprojects/yolox/datasets/dota/1.0/{dset}/images"
    IMG_DIR_2 = f"/home/av/gitprojects/yolox/datasets/dota/2.0/{dset}/images"
    out_dir = "/home/av/gitprojects/yolox/datasets/dota/coco-format/"
    out_path = out_dir+f'/annotations/{dset}.json'


    # metas = {'V1': [], 'V2': [], 'None': []}




    
    ann_id = 0




    # from avcv.debug import make_mini_coco





    mmcv.mkdir_or_exist(out_dir+'/images')
    mmcv.mkdir_or_exist(out_dir+'/annotations')

    images = []
    annotations = []
    for image_id, txt_path in tqdm(enumerate(txt_paths)):
        name = get_name(txt_path)
        p1_q = IMG_DIR_1+"/"+name+".jpg"
        p2_q = IMG_DIR_2+"/"+name+".jpg"

        im_path = p1_q if os.path.exists(p1_q) else p2_q

        # im_h, im_w = cv2.imread(im_path).shape[:2]
        width, height = imagesize.get(im_path)
        sl = out_dir+'/images/'+name+'.jpg'
        if not os.path.exists(sl):
            os.symlink(os.path.abspath(im_path), sl)

        image = dict(
            file_name=name+'.jpg',
            height=height,
            width=width,
            id=image_id
        )
        images.append(image)

        lines = open(txt_path, 'r').readlines()
        for line in lines:
            ann_id += 1
            ann = line_to_ann(line, image_id, ann_id)
            annotations.append(ann)


    mmcv.dump(dict(
        images=images,
        annotations=annotations,
        categories=cat2id.tolist()
    ), out_path)


f('train')
f('val')
