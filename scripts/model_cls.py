import os, os.path as osp
from yolox.exp.build import get_exp_by_file
import torch.nn as nn, torch

cur_dir = osp.dirname(__file__)+'/../'
class SimpleCLS2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1014, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = x[...,4:5]*x[...,5:]
        x1 = x[:, 0:2704].reshape([-1, 52, 52, 6]).permute([0,3,1,2])
        x2 = x[:, 2704:3380].reshape([-1, 26, 26, 6]).permute([0,3,1,2])
        x3 = x[:, 3380:3549].reshape([-1, 13, 13, 6]).permute([0,3,1,2])

        x1 = nn.functional.max_pool2d(x1, 4)
        x2 = nn.functional.max_pool2d(x2, 2)
        fuse = x1+x2+x3
        fuse = fuse.flatten(1)
        return self.layers(fuse)

def create_classifier(ckpt_path=f'{cur_dir}/lightning_logs/simple_nn/39/ckpts/epoch=7-val_acc=0.73.ckpt'):
    ckpt = torch.load(ckpt_path)
    st = dict()
    for k, v in ckpt['state_dict'].items():
        k = k[6:]
        st[k] = v
    model = SimpleCLS2D()
    res = model.load_state_dict(st, strict=True)
    print(res)
    return model


def create_yolox_mb2(exp_path=f'{cur_dir}/exps/dms/mb2_face_food.py', ckpt_file = f"{cur_dir}/YOLOX_outputs/mb2_face_food/best_ckpt.pth"):
    exp = get_exp_by_file(exp_path)
    model = exp.get_model()
    model.eval().cpu()
    from fvcore.nn import FlopCountAnalysis
    from collections import defaultdict 
    flops = FlopCountAnalysis(model, (torch.randn(1, 3, 256,320).to('cpu'),))
    
    ckpt = torch.load(ckpt_file, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    res = model.load_state_dict(ckpt, strict=1)
    print(res)

    from yolox.utils import get_model_info, replace_module
    from yolox.models.network_blocks import SiLU
    
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    
    model.cpu()
    return model

mb2_yolox = create_yolox_mb2()
classifier = create_classifier()

class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.mb2_yolox = mb2_yolox
        self.classifier = classifier
    def forward(self, img):
        img = img.permute([0,3,1,2])
        x = self.mb2_yolox(img)
        x = self.classifier(x)
        return x#.softmax(1)


model_wraper = ModelWrapper()

# print(model_wraper)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('out_path')
args = parser.parse_args()
torch.onnx.export(model_wraper,
                    torch.randn(1, 416, 416, 1),
                    args.out_path,
                    export_params=True,
                    opset_version=10,
                    # do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output'])

print('->', osp.abspath(args.out_path))