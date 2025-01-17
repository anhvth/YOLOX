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
        self.no_aug_epochs = 5
        self.eval_interval = 1