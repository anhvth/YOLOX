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



def video_to_coco(
    input_video,
    test_json,
    output_dir=None,
    skip=1,
    rescale=None,
    recursive=False,
):

    assert os.path.exists(input_video), f'{input_video} does not exist'
    try:
        fps = mmcv.VideoReader(input_video).fps
    except:
        fps = None

    def path2image(path, root_dir):
        w, h = Image.open(path).size
        name = path.replace(root_dir, '')
        if name.startswith('/'):
            name = name[1:]
        return dict(
            file_name=name, height=h, width=w
        )


    if output_dir is None:
        name  = get_name(input_video) if not osp.isdir(input_video) else \
            os.path.normpath(input_video).split('/')[-1]
        output_dir = osp.join('.cache/video_to_coco', name)
        logger.info(f'Set output dir to->{output_dir}')

    image_out_dir = osp.join(output_dir, 'images')

    image_dir_name = osp.normpath(image_out_dir).split('/')[-1]
    path_out_json = osp.join(output_dir, f'annotations/{image_dir_name}.json')

    mmcv.mkdir_or_exist(osp.dirname(path_out_json))

    source_type = 'dir' if osp.isdir(input_video) else 'video'

    if not osp.isdir(input_video): # IF VIDEO => EXTRACT IMAGES to image_out_dir
        mmcv.mkdir_or_exist(image_out_dir)
        video = mmcv.VideoReader(input_video)


        if rescale is not None:
            _im = mmcv.imrescale(video[0], rescale)
            im_h, im_w = _im.shape[:2]#int(rescale*im_h), int(rescale*im_w)
        else:
            im_h, im_w = video[0].shape[:2]

        is_done_extracted =  len(glob(osp.join(image_out_dir, '*.jpg'))) == len(video)
        if not is_done_extracted:
            logger.info(f'Generating images from {source_type}: {input_video} ->  {osp.abspath(output_dir)}')
            mmcv.mkdir_or_exist(image_out_dir)
            cmd = f"ffmpeg  -i {input_video} -s {im_w}x{im_h} {image_out_dir}/%06d.png"
            logger.info(f'Running command: {cmd}')
            os.system(cmd)
            to_jpg(image_out_dir)
        else:
            logger.info('Skip extracting video')

    else:
        # IF FOLDER THEN CREATE SYMLINK
        logger.info(f'Symn link {input_video}-> {image_out_dir}')
        os.symlink(osp.abspath(input_video), osp.abspath(image_out_dir))

    paths = list(sorted(glob(osp.join(image_out_dir, '*.jpg'), recursive=recursive)))
    paths += list(sorted(glob(osp.join(image_out_dir, '*.png'), recursive=recursive)))
    out_dict = dict(images=[], annotations=[], meta=dict(fps=fps),
                    categories=mmcv.load(test_json)['categories'])
    out_dict['images'] = list(
        map(partial(path2image, root_dir=image_out_dir), paths))

    for i, image in enumerate(out_dict['images']):
        image['id'] = i
    mmcv.dump(out_dict, path_out_json)
    return os.path.normpath(path_out_json), os.path.normpath(image_out_dir)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument('-i', '--json_or_mp4')
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
        '-o','--out_file', default=None, type=str, help='Path to save prediction output'
    )
    parser.add_argument(
        '--input_video', default=None, help='From video create coco format dataset'
    )
    parser.add_argument(
        '--json_test', default=None, help='Use with input video'
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
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size,exp.input_channel)))
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
        model.load_state_dict(ckpt["model"], strict=False)
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

    evaluator.get_prediction(model,
                            distributed=is_distributed,
                            half=False,
                            trt_file=None,
                            decoder=decoder,
                            test_size=exp.test_size, out_file=args.out_file)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    if args.json_or_mp4.endswith('.json'):
        args.json_path = args.json_or_mp4
    else:
        args.input_video = args.json_or_mp4 

    if args.json_path is not None:
        json_path = args.json_path
        exp.data_dir = args.json_path.split('annotations')[0]
        exp.val_ann = os.path.basename(args.json_path)
        exp.val_name = 'images'

    if args.input_video is not None:
        assert args.json_test is not None
        from avcv.coco import video_to_coco
        if osp.isdir(args.json_or_mp4):
            args.json_or_mp4 = osp.normpath(args.json_or_mp4)
            output_dir = '.cache/folder_to_coco/{}'.format(osp.basename(args.json_or_mp4))
            if osp.exists(output_dir):
                os.system(f'rm -r "{output_dir}"')

        else:
            output_dir = osp.join(osp.dirname(args.json_or_mp4), osp.basename(args.json_or_mp4).split('.')[0])
        json_path = video_to_coco(args.json_or_mp4, args.json_test, output_dir, rescale=(640, 640))[0]
        exp.data_dir = json_path.split('annotations')[0]
        exp.val_ann = os.path.basename(json_path)
        exp.val_name = 'images'
        exp.annotation_dir = 'annotations'
        args.dump = True

    if args.out_file is None:
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
