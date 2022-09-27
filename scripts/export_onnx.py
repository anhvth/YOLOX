import os, os.path as osp
from yolox.exp.build import get_exp_by_file
import torch.nn as nn, torch
from fastcore.all import *
from loguru import logger

CLASSES = ['face', 'eye', 'mouth', 'phone', 'cigarette', 'food/drink']
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
            nn.Conv2d(6,32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 4, (1,3), padding=0),
        )

    def forward(self, x):
        if self.training:
            return self.layers(x)[:,:,0,0]
        else:
            return self.layers(x)





def create_yolox_mb2(exp_path=f'{cur_dir}/exps/dms/mb2_face_food.py', ckpt_file = f"{cur_dir}/YOLOX_outputs/mb2_face_food/best_ckpt.pth"):
    exp = get_exp_by_file(exp_path)
    model = exp.get_model()
    model.eval().cpu()
    
    ckpt = torch.load(ckpt_file, map_location="cpu")
    ckpt = ckpt["model"]
    res = model.load_state_dict(ckpt, strict=1)
    logger.info(res)
    
    return model.cpu().eval().requires_grad_(False), ckpt

mb2_yolox, ckpt = create_yolox_mb2()
@patch
def forward_classification(self:type(mb2_yolox.head), xin, labels=None, imgs=None):
    outputs = []
    origin_preds = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []

    for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
        zip(self.cls_convs, self.reg_convs, self.strides, xin)
    ):
        x = self.stems[k](x)
        cls_x = x
        reg_x = x

        cls_feat = cls_conv(cls_x)
        cls_output = self.cls_preds[k](cls_feat)

        reg_feat = reg_conv(reg_x)
        reg_output = self.reg_preds[k](reg_feat)
        obj_output = self.obj_preds[k](reg_feat)


        output = torch.cat(
            [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
        )
        outputs.append(output)
    self.hw = [x.shape[-2:] for x in outputs]
    new_outputs = []
    pool_sizes = [4,2,1]
    for i in range(3):
        pool_size = pool_sizes[i]
        output = outputs[i]
        output = output[:,4:5]*output[:,5:]
        new_f = nn.functional.max_pool2d(output, pool_size)
        new_outputs.append(new_f)
    return sum(new_outputs)

class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = SimpleCLS2D()

    def forward(self, img):
        with torch.no_grad():
            mb2_yolox.to(img.device)
            x = mb2_yolox.backbone(img)
            x = mb2_yolox.head.forward_classification(x)
        x = self.classifier(x)
        return x




def load_ckpt(model_wraper, ckpt_path):
    st = torch.load(ckpt_path)['state_dict']
    new_st = {}
    for k, v in st.items():
        new_st[k[6:]] = v
    res = model_wraper.load_state_dict(new_st)
    print('model_wraper load:', res)
    return model_wraper


if __name__ == '__main__':
    # print(model_wraper)
    model_wraper = ModelWrapper()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('out_path')
    args = parser.parse_args()
    load_ckpt(model_wraper, args.ckpt)
    torch.onnx.export(model_wraper,
                        torch.randn(1, 1, 224, 416),
                        args.out_path,
                        export_params=True,
                        opset_version=10,
                        # do_constant_folding=True,
                        input_names = ['input'],
                        output_names = ['output'])

    print('->', osp.abspath(args.out_path))