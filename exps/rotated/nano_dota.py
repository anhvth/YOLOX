#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp
import torch.distributed as dist
import mmcv


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 18
        self.depth = 0.33
        self.input_size = (640, 640)
        self.width = 0.25
        self.scale = (0.5, 1.5)
        self.random_size = None  # (10, 20)
        self.test_size = (640, 640)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        self.print_interval = 5

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, RotatedYOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = RotatedYOLOXHead(self.num_classes, self.width,
                                    in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, dataset=None):
        from yolox.data import (
            RotatedCOCODataset,
            DataLoader,
            InfiniteSampler,
            RotTrainTransform,
            YoloBatchSampler
        )
        from avcv.utils import TimeLoger

        time_dict = TimeLoger()
        dataset = RotatedCOCODataset(
            data_dir='datasets/dota/coco-format/',
            json_file='train.json',
            name='images',
            # data_dir='datasets/dota/coco-format/mini-debug',
            # json_file='mini_json.json',
            # name='images',            

            preproc=RotTrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=50,
                accepted_min_box_size=-1,
            )
        )

        assert self.num_classes == len(dataset.coco.dataset['categories']), "{}, {}".format(
            self.num_classes, len(dataset.coco.dataset['categories'])
        )

        time_dict.update('create dataset')

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        time_dict.update('build_sampler')
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        time_dict.update('build_batch_sampler')


        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        time_dict.update('build_train_loader')
        print(time_dict)

        return train_loader
