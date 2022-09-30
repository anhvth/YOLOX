
# Generate high entropy images dataset in coco format
# from yolox.utils import 
import argparse
import os
from loguru import logger
import mmcv
import numpy as np
from avcv.coco import CocoDataset
import os
import os.path as osp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('task', choices=['dump', 'stitch'])
    parser.add_argument('gt')
    parser.add_argument('--imgs', default=None)
    parser.add_argument('--pred', default=None)
    parser.add_argument('--is_video', default=False, action='store_true')
    parser.add_argument('--num_samples', '-n',default=100)
    parser.add_argument('--score_thr', default=0.3, type=float)
    parser.add_argument('--out_dir', default='.cache/visualize_samples/')
    parser.add_argument('--output_size', default='500,300')

    args = parser.parse_args()
    dc = CocoDataset(args.gt, args.imgs, args.pred)
    try:
        dataset_name = args.gt.split('/')[-3]
    except:
        dataset_name = 'dataset_name'

    def visualize_one_frame(img_id):
        img = dc.visualize(img_id, score_thr=args.score_thr)
        if args.is_video:
            return img
        else:
            filename = osp.basename(dc.gt.imgs[img_id]['file_name'])
            out_path = os.path.join(args.out_dir, dataset_name, filename)
            logger.info(out_path)
            mmcv.imwrite(img,out_path)
    if args.is_video:
        logger.info('Visualize video')
        from avcv.all import *        
        imgs = multi_thread(visualize_one_frame, dc.img_ids)
        mmcv.mkdir_or_exist(args.out_dir)
        output_size = [int(_) for _ in args.output_size.split(',')]
        images_to_video(imgs[0::3], osp.join(args.out_dir, 'vis_video.mp4'), output_size=output_size)
    else:
        for img_id in np.random.choice(dc.img_ids, args.num_samples, replace=False):
            visualize_one_frame(img_id)
    
