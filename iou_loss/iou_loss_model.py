import torch
import os
import mmcv
from tqdm import tqdm
import torch
import torchvision
from torch import nn
import numpy as np
import cv2
from avcv.visualize import plot_images

MASK_SIZE = 64 # the size at which we assume
PAD_SIZE = MASK_SIZE//4    # buffer 
TRAIN_SIZE = 64 # return
MAX_ANGLE=90

class IoULossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, 1024))
        
        self.embed =  nn.Linear(5, 1024)
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        

        
    def forward(self, x):
        x = self.embed(x) #[bz,64]
        x = self.encode(x)
        x = x.reshape([-1, 1, 32, 32])
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = nn.functional.interpolate(x, scale_factor=2) 
        
        out2 = self.conv_out(x)
        
        return out2.sigmoid()

    

def coco_segmentation_to_rbox(segmentation):
    x = segmentation
    x = np.array(x).reshape([-1,2]).astype(int)
    return cv2.minAreaRect(x)


def normalize_input(rbox, im_w, im_h):
    (cx,cy),(w,h), a = rbox
    return np.array([cx/im_w, cy/im_h, w/im_w, h/im_h, a/MAX_ANGLE])

class InputTarget:

    def __init__(self):        
        self._mask_padding = np.zeros([MASK_SIZE+PAD_SIZE*2, \
                                       MASK_SIZE+PAD_SIZE*2], dtype='uint8')
    
    @property
    def input(self):
        return np.array([self.cx, self.cy, self.w, self.h, self.a]).astype(np.float32)
    
    def set_input(self, input, normalize=False):
        cx,cy,w,h,a = input
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.a = a
    
    
    def get_box(self):
        cx,cy,w,h = self.input[:4]*MASK_SIZE
        a = self.input[-1]*MAX_ANGLE
        cx += PAD_SIZE
        cy += PAD_SIZE
        return cx,cy,w,h,a
    
    @property
    def target(self):
        cx,cy,w,h,a = self.get_box()
        points = cv2.boxPoints(((cx,cy), (w,h), a))
        points = np.array(points).astype(int)
        assert np.min(points) > 0, points
        mask = self._mask_padding.copy()
        cv2.drawContours(mask, [points], -1, 1, -1)
        return cv2.resize(mask, (TRAIN_SIZE, TRAIN_SIZE))
    
    def show(self, dpi=None, img=None):
        cx,cy,w,h,a = self.get_box()
        points = cv2.boxPoints(((cx,cy), (w,h), a))
        points = np.array(points).astype(int)
        assert np.min(points) > 0, points
        mask = self._mask_padding.copy()
        mask = np.stack([mask]*3, -1).astype('uint8')
        
        
        if img is not None:
            img = mmcv.imread(img)
            img = cv2.resize(img, (MASK_SIZE, MASK_SIZE))
            mask[PAD_SIZE:PAD_SIZE+MASK_SIZE, PAD_SIZE:PAD_SIZE+MASK_SIZE] = img
        cv2.drawContours(mask, [points], -1, (255,0,0), 1)
        
        mask= cv2.rectangle(mask, (PAD_SIZE, PAD_SIZE), (PAD_SIZE+MASK_SIZE, PAD_SIZE+MASK_SIZE), (0, 255, 0), 1)
        
        
        mask = cv2.resize(mask, (256, 256))
        cv2.putText(mask, "{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}".format(*self.input.tolist()), (22,22), 
                    cv2.FONT_HERSHEY_SIMPLEX, .5,
                       (255,0,0), 1)           
        
        if dpi is not None:
            print("Training at size:", self.target.shape)
            plot_images([mask, self.target], mxn=[1,2], dpi=200)
        return mask