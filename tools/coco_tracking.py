from loguru import logger

import cv2
import numpy as np

import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

import argparse
import os
import os.path as osp
import pandas as pd
import time
from avcv.all import *


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("path",
                        default="./videos/palace.mp4",
                        help="path to images or video")
    parser.add_argument("--output_dir", default="./results_vis")

    parser.add_argument(
        "--visualize",
        '-v',
        action="store_true",
        default=False,
        help="whether to save the inference result of image/video",
    )
    parser.add_argument('--categories', default='1,77')
    parser.add_argument("--track_thresh",
                        type=float,
                        default=0.1,
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer",
                        type=int,
                        default=3000,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh",
                        type=float,
                        default=1.1,
                        help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=10,
        help=
        "threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area',
                        type=float,
                        default=10,
                        help='filter out tiny boxes')
    parser.add_argument('--csv_out_path',default=None)
    parser.add_argument("--no-verbose",
                        dest="verbose",
                        default=True,
                        action="store_false",
                        help="verbose.")
    return parser


def get_bboxes(anns,
               mode='xywh',
               dtype=np.float32,
               score_thr=None,
               category_ids=[0]):
    bboxes = []
    for ann in anns:
        if not ann['category_id'] in category_ids: continue
        if score_thr is not None and ann.get('score', False):
            if ann['score'] < score_thr:
                continue
        x, y, w, h = ann['bbox']
        if mode == 'xywh':
            bboxes.append([x, y, w, h, ann['score']])
        elif mode == 'xyxy':
            bboxes.append([x, y, x + w, y + h, ann['score']])
        elif mode == 'cxcywh':
            cx = x + w / 2
            cy = y + h / 2
            bboxes.append([cx, cy, w, h, ann['score']])
        else:
            raise NotImplemented
    bboxes = np.array(bboxes)
    if dtype is not None:
        bboxes = bboxes.astype(dtype)
    return bboxes


class MyCocoDataset(CocoDataset):
    def __init__(self, gt, im, pred, **kwargs):
        super().__init__(gt, im, pred, **kwargs)
        img_dir = osp.join(osp.dirname(im), get_name(im), 'images')
        # import ipdb; ipdb.set_trace()
        if osp.exists(img_dir):
            self.img_dir = img_dir
        elif osp.isfile(im):
            self.video = mmcv.VideoReader(im)

    def imread(self, img_id):
        if hasattr(self, 'video'):
            if img_id >= len(self.video):
                logger.warning(
                    f"Index {img_id} out of video range: {len(self.video)}")
                return None
            return self.video[img_id]

        return super().imread(img_id)


def get_csv_out_path(path):
    out_dir = osp.dirname(path)
    name = get_name(args.path)
    res_file = osp.join(out_dir, f'{name}/bytetrack.csv')
    return res_file


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def get_coco_pred(path):
    ext = osp.basename(path).split('.')[-1]
    ann_path = path.replace(f".{ext}", "/annotations/images.json")
    pred_path = path.replace(f".{ext}", "/annotations/pred.json")
    assert osp.exists(ann_path), ann_path
    assert osp.exists(pred_path), pred_path
    cc = MyCocoDataset(ann_path, path, pred=pred_path)
    fn2id = {
        int(get_name(img['file_name'])): img['id']
        for img in cc.gt.imgs.values()
    }
    ks = list(cc.gt.imgs.keys())
    return cc


def imageflow_demo(args, categories=[1], csv_out_path=None):
    csv_out_path = csv_out_path if csv_out_path is not None else get_csv_out_path(args.path)
    save_folder = osp.join(args.output_dir, get_name(args.path))
    save_path = osp.join(save_folder,
                         "{}_track_vis.mp4".format(get_name(args.path)))

    os.makedirs(args.output_dir, exist_ok=True)
    cc = get_coco_pred(args.path)
    pred = cc.pred
    os.makedirs(save_folder, exist_ok=True)

    tracker = {cat: BYTETracker(args, frame_rate=30) for cat in categories}
    results = [f"img_id,tid,cat_id,x,y,w,h,score\n"]
    vis_imgs = []
    img_ids = list(pred.imgs.keys())
    imsize = None

    tracklets = dict()
    pbar = tqdm(sorted(img_ids))

    img_info = pred.imgs[img_ids[0]]
    imsize = (img_info['height'], img_info['width'])

    for img_id in pbar:
        frame_id = img_id
        ret_val = True
        anns = pred.imgToAnns[img_id]
        if args.visualize:
            online_im = mmcv.imresize(cc.imread(img_id), imsize[::-1])

        for category_id in categories:
            outputs = torch.from_numpy(
                get_bboxes(anns,
                           'xyxy',
                           category_ids=[category_id],
                           score_thr=0.7))
            # print(len(outputs))
            if outputs is not None and len(outputs) > 0:
                online_targets = tracker[category_id].update(
                    outputs, imsize, imsize)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{category_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                        )

                if args.visualize:
                    online_im = plot_tracking(online_im,
                                              online_tlwhs,
                                              online_ids,
                                              frame_id=frame_id + 1,
                                              fps=-1)
        if args.visualize:
            vis_imgs.append(online_im)
        frame_id += 1

    
    with open(csv_out_path, 'w') as f:
        f.writelines(results)

    logger.info(f"save results to {csv_out_path}")

    if args.visualize:
        images_to_video(vis_imgs,
                        save_path,
                        output_size=imsize[::-1],
                        verbose=False)
        print(
            f'rs dms:"{osp.abspath(save_path)}" ./ && open "{osp.basename(save_path)}"'
        )
    return csv_out_path

def track_coco(cc, args, categories=[1, 77], verbose=False):
    save_folder = osp.join(args.output_dir, args.name)
    save_path = osp.join(save_folder, "{}_track_vis.mp4".format(args.name))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    tracker = {cat:BYTETracker(args, frame_rate=30) for cat in categories}
    results = []
    vis_imgs = []
    imgs = list(sorted(cc.gt.imgs.values(), key=lambda img:img['file_name']))
    img_ids = [img['id'] for img in imgs]
    imsize = None
    tracklets = dict()
    img_info = cc.gt.imgs[img_ids[0]]  
    imsize = (img_info['height'], img_info['width'])
    # for img_id in :
    if verbose:
        pbar = tqdm(img_ids)
    else:
        pbar = img_ids
    for img_id in pbar:
        frame_id = img_id
        ret_val = True
        anns = cc.gt.imgToAnns[img_id]
        for category_id in categories:
            outputs = torch.from_numpy(get_bboxes(anns, 'xyxy', category_ids=[category_id], score_thr=0.1))
            if outputs is not None and len(outputs) > 0:
                online_targets = tracker[category_id].update(outputs, imsize, imsize)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        row_data = [frame_id,tid,category_id,tlwh[0],tlwh[1],tlwh[2],tlwh[3],t.score.item(), t]
                        results.append(row_data)

        frame_id += 1

    df = pd.DataFrame(data=results, columns=['img_id','tid','cat_id','x','y','w','h','score', 't'])
    return df

def run_track(cc, name, verbose=0, args=None):
    if args is None:
        parser = make_parser()
        args = parser.parse_known_args()[0]
        args.save_result = True
        args.mot20 = False
        
    default_track_args = args
    #--------------------------
    default_track_args.name = name
    return track_coco(cc, default_track_args, [1], verbose=verbose)
def visualize_track_df(cc, df, name):
    img_ids = sorted(cc.img_ids, key=lambda img_id: cc.gt.imgs[img_id]['file_name'])
    vis_imgs = []
    logger.info('Visualize tracking')
    # for frame_id, img_id in tqdm(enumerate(img_ids), total=len(cc.img_ids)):
    def f_iter(inp):
        frame_id, img_id = inp
        img = cc.gt.imgs[img_id]
        tracks = df[df.img_id==img_id]
        is_track = tracks['t'].apply(lambda t:t.is_activated)
        online_tlwhs = tracks[['x', 'y', 'w', 'h']].values
        
        online_tlwhs = [item for item, tracked in zip(online_tlwhs, is_track) if tracked]
        
        online_ids = tracks['tid'].values
        online_im = cc.imread(img_id)
        online_im = plot_tracking(online_im,
                                  online_tlwhs,
                                  online_ids,
                                  frame_id=frame_id,
                                  fps=-1)
        return online_im
    
    vis_imgs = multi_thread(f_iter, enumerate(img_ids))
    save_path = 'results_vis/{}.mp4'.format(name)
    images_to_video(vis_imgs, save_path, output_size=(480, 320), verbose=False)
    print(f'rs dms:"{osp.abspath(save_path)}" ./ && open "{osp.basename(save_path)}"')
    
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_known_args()[0]
    args.mot20 = False
    categories = [int(_) for _ in args.categories.split(',') if _.isdigit()]
    logger.info(f'Tracking ids: {categories}')
    
    csv_out_path = get_csv_out_path(args.path) if args.csv_out_path is None else args.csv_out_path

    imageflow_demo(args, categories, csv_out_path=csv_out_path)


