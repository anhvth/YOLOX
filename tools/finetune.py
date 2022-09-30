#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


from tools.train import make_parser

def get_finetuner(exp, args):
    from yolox.core.trainer import Finetuner
    finetuner = Finetuner(exp, args)
    # NOTE: finetuner shouldn't be an attribute of exp object
    return finetuner
@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    finetuner = get_finetuner(exp, args)
    finetuner.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    if args.debug:
        exp.data_num_workers = 0
        args.devices = 1
        exp.train_ann = exp.val_ann
        exp.train_name = exp.val_name
        exp.batch_size = 2
        exp.print_interval = 1

    if args.vis_batches:
        exp.data_num_workers = 0
        exp.batch_size = 4
        args.devices = 1
        exp.eval_interval = 10000
        
    if args.finetune is not None:
        assert args.ckpt is not None
        exp.finetune = args.finetune

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
