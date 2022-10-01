import os
from exps.dms.food_m import Exp as FoodMExp

class Exp(FoodMExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Generated from nbs/concat_food_dring_smoking.ipynb
        self.train_ann = "train_3class_phone_cigarette_food_with_coco.json" 

        # Schedual
        self.max_epoch = 10
        self.warmup_epochs = 2
        self.no_aug_epochs = 3
        self.eval_interval = 1

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