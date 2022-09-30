import os
from exps.dms.mb2_food import Exp as FoodMb2Exp

class Exp(FoodMb2Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Generated from nbs/concat_food_dring_smoking.ipynb
        self.train_ann = "train_3class_phone_cigarette_food_with_coco.json" 

        # Schedual
        self.max_epoch = 10
        self.warmup_epochs = 2
        self.eval_interval = 1
        self.no_aug_epochs = 3