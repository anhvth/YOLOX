import os, os.path as osp
from yolox.exp.build import get_exp_by_file
import torch.nn as nn, torch

def get_M(input_size = [224, 416], num_cls=6):
    """
        return list of mapping [old_start, old_end, new_start, new_end ]
    """
    M = []
    target_f_size = []
    
    cur_i = 0
    o_cur_i = 0

    for i in [8,16,32]:
        h, w = input_size
        _h, _w = h//i, w//i
        _oh = _ow = 416//i

        a, b = cur_i,cur_i +_h*_w
        oa, ob = o_cur_i,o_cur_i +_oh*_oh
        cur_i = b
        o_cur_i = ob
        M.append([oa, ob, _h*_w, _h, _w])
        
    return M, _h*_w*num_cls

M, IN_CHANNEL_CLS = get_M()
print(f'{IN_CHANNEL_CLS=}')
NUM_CLASSES = 6

cur_dir = osp.dirname(__file__)+'/../'
class SimpleCLS2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(IN_CHANNEL_CLS, 256),
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
        xs = []
        cur_i = 0
        for i, _ in enumerate(M):
            h,w = _[-2:]
            _x = x[:, cur_i:cur_i+h*w].reshape([-1, h, w, 6]).permute([0,3,1,2])
            
            pool_size = 2**(2-i)
            _x = nn.functional.max_pool2d(_x, pool_size)
            cur_i += h*w
            xs.append(_x)
            # print(_x.shape)
        fuse = sum(xs).flatten(1)
        return self.layers(fuse)

def create_classifier(ckpt_path=f'{cur_dir}/lightning_logs/simple_nn/00/ckpts/epoch=6-val_acc=0.74.ckpt'):
    ckpt = torch.load(ckpt_path)
    st = dict()
    for k, v in ckpt['state_dict'].items():
        k = k[6:]
        st[k] = v
    model = SimpleCLS2D()
    res = model.load_state_dict(st, strict=False)
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
                    torch.randn(1, 224, 416, 1),
                    args.out_path,
                    export_params=True,
                    opset_version=10,
                    # do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output'])

print('->', osp.abspath(args.out_path))