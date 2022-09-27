#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger


import argparse
import os
import os.path as osp
import random
import warnings


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--not_replace",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--eval_trainset",
        default=False,
        action="store_true",
        help="Evaluating on train set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        '--folder',
        default=None,
        help="Load model and run detection for a given model without evaluating then dump the detection using COCOFormat"
        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        '--json_path', default=None
    )
    parser.add_argument(
        '--out_file', default=None, type=str, help='Path to save prediction output'
    )
    parser.add_argument(
        '--input_video', default=None, help='From video create coco format dataset'
    )
    parser.add_argument(
        '--json_test', default=None, help='Use with input video'
    )
    parser.add_argument(
        '--out_dir', default=None, help='Out coco format dir'
    )
    parser.add_argument('--dump', action='store_true', default=False)

    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)
    # setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size, exp.input_channel)))
    # logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"], strict=True)
        # print(model)
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None
    
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, out_file=args.out_file,
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    if args.ckpt is None:
        args.ckpt = os.path.join(f'YOLOX_outputs/{exp.exp_name}/best_ckpt.pth')
    if args.json_path is not None:
        json_path = args.json_path
        exp.data_dir = args.json_path.split('annotations')[0]
        exp.val_ann = os.path.basename(args.json_path)
        if args.json_test is not None:
            exp.json_test = args.json_test

    if args.input_video is not None:
        # assert args.json_test is not None
        if args.json_test is None:
            args.json_test = osp.join(exp.data_dir, 'annotations', exp.val_ann)
        from avcv.all import video_to_coco, get_name
        out_coco_format_dir = osp.dirname(args.input_video) if args.out_dir is None else args.out_dir #osp.join(, get_name(args.input_video))
        output_dir = osp.join(out_coco_format_dir, osp.basename(args.input_video).split('.')[0])
        args.out_file = osp.join(output_dir, 'annotations', f'pred_{exp.exp_name}.json')
        json_path = video_to_coco(args.input_video, args.json_test, output_dir, rescale=(512, 512))[0]
        exp.data_dir = json_path.split('annotations')[0]
        exp.val_ann = os.path.basename(json_path)
        exp.val_name = 'images'
        exp.annotation_dir = 'annotations'
        args.dump = True

    if args.eval_trainset:
        exp.val_ann = exp.train_ann
        exp.val_name = exp.train_img_dir
    if args.dump and args.out_file is None:
        args.out_file = osp.join(osp.dirname(json_path), 'pred.json')
        logger.info('out prediction is set to: {}'.format(osp.abspath(args.out_file)))

    if args.not_replace and osp.exists(args.out_file):
        logger.info(f'{args.out_file} exits, process stoped')
    else:
        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
        assert num_gpu <= torch.cuda.device_count()

        dist_url = "auto" if args.dist_url is None else args.dist_url
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args, num_gpu),
        )
