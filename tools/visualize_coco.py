
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
    parser.add_argument('--num_samples', '-n',default=100)
    parser.add_argument('--out_dir', default='.cache/visualize_samples/')

    args = parser.parse_args()
    dc = CocoDataset(args.gt, args.imgs, args.pred)
    dataset_name = args.gt.split('/')[-3]

    for img_id in np.random.choice(dc.img_ids, args.num_samples, replace=False):
        img = dc.visualize(img_id, mode='gt' if args.pred is None else 'pred')
        filename = osp.basename(dc.gt.imgs[img_id]['file_name'])
        out_path = os.path.join(args.out_dir, dataset_name, filename)
        logger.info(out_path)
        mmcv.imwrite(img,out_path)
