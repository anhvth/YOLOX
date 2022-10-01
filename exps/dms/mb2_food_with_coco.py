import os
from exps.dms.mb2_food import Exp as FoodMb2Exp

class Exp(FoodMb2Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Generated from nbs/concat_food_dring_smoking.ipynb
        self.train_ann = "train_3class_phone_cigarette_food_with_coco.json" 
        #data
        self.data_num_workers = 4
        # Schedual
        self.max_epoch = 150
        self.warmup_epochs = 5
        self.eval_interval = 5
        self.no_aug_epochs = 15

        #-
        self.num_classes = 3
        
    def get_finetune_data_loader(self, *args, **kwargs):
        from loguru import logger
        from avcv.all import mmcv, identify, osp
        json_path = self.dataset._dataset.json_path
        data = mmcv.load(json_path)
        new_imgs = []
        for img in data['images']:
            if not '/coco/' in img['file_name']:
                new_imgs.append(img)
        data['images'] = new_imgs
        tmp_path = osp.join('/tmp/{}.json'.format(identify(new_imgs)))
        if not osp.exists(tmp_path):
            mmcv.dump(data, tmp_path)
        self.train_ann = tmp_path
        logger.info('\t\t Num of images: {}'.format(len(data['images'])))
        return self.get_data_loader(*args, **kwargs)