#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 4
        self.depth = 0.67
        self.width = 0.75
        self.input_channel = 3
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # self.data_num_workers = 4
        # self.input_size = (416, 416)
        # self.multiscale_range = 5
        # self.random_size = (10, 20)
        # self.mosaic_scale = (0.5, 1.5)
        # self.test_size = (416, 416)
        # self.mosaic_prob = 0.5
        # self.hsv_prob = -1.0
        # self.enable_mixup = False
        # self.act = 'relu'
        # ('/data/full-version-vip-pro/coco_annotations/train.json', '/data/full-version-vip-pro/DMS_DB_090922/', 'face')

        self.data_dir = "/data/full-version-vip-pro/"
        # self.img_dir = dict(train='DMS_DB_090922', val='DMS_DB_090922')
        self.train_name = self.val_name = 'DMS_DB_090922'
        self.train_ann = "train.json"
        self.val_ann = "train.json"
        self.test_ann = "val.json"
        self.basic_lr_per_img = 0.005 / 64.0
        self.max_epoch = 30

        self.flip_prob = 0


    # def get_model(self, sublinear=False):
    #     super().get_model()
    #     model = self.model
    #     import ipdb; ipdb.set_trace()

    # def get_model(self, sublinear=False):
    
    #     def init_yolo(M):
    #         for m in M.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eps = 1e-3
    #                 m.momentum = 0.03
    #     if "model" not in self.__dict__:
    #         from yolox.models import YOLOX, MobilenetV2PAFPN, YOLOXHead
    #         in_channels = [32, 96, 320]
    #         # MobileNetV2 model use depthwise = True, which is main difference.
    #         backbone = MobilenetV2PAFPN(
    #             self.depth, self.width, in_channels=in_channels, first_channel=1,
    #             act=self.act, depthwise=True, pretrained=None
    #         )
    #         head = YOLOXHead(
    #             self.num_classes, self.width, in_channels=in_channels,
    #             act=self.act, depthwise=True
    #         )
    #         self.model = YOLOX(backbone, head)

    #     self.model.apply(init_yolo)
    #     self.model.head.initialize_biases(1e-2)
    #     return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCOIRDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCOIRDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_name,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCOIRDataset, ValTransform

        valdataset = COCOIRDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            json_test=self.json_test,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.AdamW(
                pg0, lr=lr, betas=(self.momentum, 0.999)
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer


if __name__ == '__main__':
    exp = Exp()
    print(exp.get_model())
    data = exp.get_data_loader(batch_size=1, is_distributed=False)

    # import ipdb; ipdb.set_trace()
    x = next(iter(data))[0]
    print(x.shape)