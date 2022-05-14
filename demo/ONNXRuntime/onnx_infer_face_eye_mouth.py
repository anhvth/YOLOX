#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import torch
import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from PIL import Image
from timm.data.transforms_factory import  transforms_imagenet_eval
COCO_CLASSES =['face', 'eye', 'mouth']

lh0, lc0, rh0, rc0, mh0, mc0 = [np.zeros([2,1,32], dtype=np.float32) for _ in range(6)]

T = transforms_imagenet_eval(32)
if not 'cnnlstm' in dir():
    cnnlstm = onnxruntime.InferenceSession('cnnlstm.onnx')

def get_inps(x, lh0, lc0, rh0, rc0, mh0, mc0):
    def _px(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
        
    ort_inputs = {cnnlstm.get_inputs()[i].name: _px(x) for i, x in enumerate([x, lh0, lc0, rh0, rc0, mh0, mc0])}
    # torch_inputs = [torch.from_numpy(x).to(device) for x in ort_inputs.values()]
    return ort_inputs #, torch_inputs

def prob_viz(res, input_frame, cls_names = ['awake', 'drowsy', 'sleepy', 'level'],
    colors = [(245,117,16), (117,245,16), (16,117,245), (0,0,255)]):
    output_frame = input_frame
    cv2.rectangle(output_frame, (0,10), (int(1*222), 40+3*40), (0,0,0), -1)
    for num, prob in enumerate(res):
        output_frame = cv2.rectangle(output_frame, (0,10+num*40), (int(prob*100), 40+num*40), colors[num], -1)
        # cv2.rectangle(output_frame, (0,60+num*40), (int(1*100), 90+num*40), (255,255,255), 1)
        output_frame = cv2.putText(output_frame, '{}-{:0.1f}%'.format(cls_names[num], prob*100), (0, 35+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame



def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def capture_n_do(do):

    # import the opencv library
    import cv2
    
    
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        out = do(frame)
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = make_parser().parse_args()
    session = onnxruntime.InferenceSession(args.model)

    def do(origin_img):
        global lh0, lc0, rh0, rc0, mh0, mc0
        input_shape = tuple(map(int, args.input_shape.split(',')))
        # origin_img = cv2.imread(image_path)
        img, ratio = preprocess(origin_img, input_shape)

        

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            # import ipdb; ipdb.set_trace()


            def crop(img, bbox, size=None):
                import mmcv
                try:
                    if bbox is None:
                        # h,w = size
                        return np.zeros([*size, 3]).astype('uint8')
                    x,y,x2,y2 = [int(_) for _ in bbox[:4]]
                    img = mmcv.imread(img).copy()
                    img = img[y:y2, x:x2]
                    if size is not None:
                        img = mmcv.imrescale(img, size[::-1])
                        img = mmcv.impad(img, shape=size)
                    return img
                except Exception as e:
                    # print(e)
                    return np.zeros([*size, 3]).astype('uint8')
            def get_crop(cls_id, n):
                eye_ids = np.where(final_cls_inds==cls_id)[0]
                eyes = final_boxes[eye_ids]
                outs = [np.zeros([32, 32]) for _ in range(n)]
                for i in range(n):
                    if i >= len(eyes):
                        break
                    c = crop(origin_img, eyes[i], (32,32))
                    outs[i]= c
                return outs
                
            # eyes = get_crop(1, 2)
            # mouth = get_crop(1, 1)
            # x = [*eyes, *mouth]
            # # for _ in x:
            #     # print(_.shape)
            # x = [T(Image.fromarray(_)) for _ in x]
            # x = np.stack(x)[None, None]

            # inps = get_inps(x, lh0, lc0, rh0, rc0, mh0, mc0)
            # output, lh0, lc0, rh0, rc0, mh0, mc0 = cnnlstm.run(None, inps)
            # output = torch.from_numpy(output).sigmoid().squeeze().numpy()
            # # import ipdb; ipdb.set_trace()
            # origin_img = prob_viz(output, origin_img)


            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=args.score_thr, class_names=COCO_CLASSES)
        return origin_img

    capture_n_do(do)
    # mkdir(args.output_dir)
    # output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    # cv2.imwrite(output_path, origin_img)
