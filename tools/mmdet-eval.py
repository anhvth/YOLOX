import argparse
import os.path as osp
import numpy as np
import mmcv
from terminaltables import AsciiTable
from mmcv.utils import print_log
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import CocoDataset, api_wrappers
from pycocotools.coco import COCO
from mmdet.datasets.api_wrappers import COCOeval
import itertools


class TSDataset(CocoDataset):
    CLASSES = ['Bien cam', 'Nguy Hiem', 'Hieu Lenh', 'Chi Dan', 'Con Lai', 'Den giao thong']


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMdet eval coco')
    parser.add_argument('output', help='test config file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gt = COCO('datasets/tsd/vinai_bdi_combined/annotations/test.json')
    output = mmcv.load(args.output)
    pred = gt.loadRes(output)
    cocoEval = COCOeval(gt, pred, 'bbox')
    # cocoEval.params.iouThrs = np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
    cocoEval.params.iouThrs = np.array([0.5])
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    catIds = cocoEval.params.catIds

    precisions = cocoEval.eval['precision']
    
    # precision: (iou, recall, cls, area range, max dets)
    # assert len(self.cat_ids) == precisions.shape[2]

    results_per_category = []
    for idx, catId in enumerate(catIds):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = gt.cats[idx]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        results_per_category.append(
            (f'{nm["name"]}', f'{float(ap):0.3f}'))

    num_columns = min(2, len(results_per_category) * 2)
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['category', 'AP'] * (num_columns // 2)
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print_log('\n' + table.table)


if __name__ == '__main__':
    main()
