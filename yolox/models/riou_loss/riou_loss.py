
import torch
import mmcv
import torch.nn
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import cv2
from avcv.visualize import plot_images

MASK_SIZE = 64  # the size at which we assume
PAD_SIZE = MASK_SIZE//4    # buffer
TRAIN_SIZE = 64  # return
MAX_ANGLE = 90


class IoULossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, 1024))

        self.embed = nn.Linear(5, 1024)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.embed(x)  # [bz,64]
        x = self.encode(x)
        x = x.reshape([-1, 1, 32, 32])

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = nn.functional.interpolate(x, scale_factor=2)

        out2 = self.conv_out(x)

        return out2.sigmoid()


def coco_segmentation_to_rbox(segmentation):
    x = segmentation
    x = np.array(x).reshape([-1, 2]).astype(int)
    return cv2.minAreaRect(x)


def normalize_input(rbox, im_w, im_h, mode='rbox'):
    assert mode in ['rbox', 'batch_rbox']
    if mode == 'rbox':
        if isinstance(rbox, tuple):
            (cx, cy), (w, h), a = rbox
        else:
            cx, cy, w, h, a = rbox
        return np.array([cx/im_w, cy/im_h, w/im_w, h/im_h, a/MAX_ANGLE])
    elif mode == 'batch_rbox':
        rbox = np.array([normalize_input(_, im_w, im_h, 'rbox') for _ in rbox])
        return rbox


class InputTarget:
    def __init__(self):
        self._mask_padding = np.zeros([MASK_SIZE+PAD_SIZE*2,
                                       MASK_SIZE+PAD_SIZE*2], dtype='uint8')

    @property
    def input(self):
        input = np.array([self.cx, self.cy, self.w, self.h, self.a]).astype(np.float32)
        input = np.clip(input, 0, 2)
        assert np.logical_and(input >= 0, input <= 2).all()
        return input

    def set_input(self, input, normalize=False):
        cx, cy, w, h, a = input
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.a = a

    def get_box(self):
        cx, cy, w, h = self.input[:4]*MASK_SIZE
        a = self.input[-1]*MAX_ANGLE
        cx += PAD_SIZE
        cy += PAD_SIZE
        return cx, cy, w, h, a

    @property
    def target(self):
        cx, cy, w, h, a = self.get_box()
        points = cv2.boxPoints(((cx, cy), (w, h), a))
        points = np.array(points).astype(int)
        assert np.min(points) > 0, points
        mask = self._mask_padding.copy()
        cv2.drawContours(mask, [points], -1, 1, -1)
        return cv2.resize(mask, (TRAIN_SIZE, TRAIN_SIZE))

    def show(self, dpi=None, img=None):
        cx, cy, w, h, a = self.get_box()
        points = cv2.boxPoints(((cx, cy), (w, h), a))
        points = np.array(points).astype(int)
        # assert np.min(points) > 0, points
        mask = self._mask_padding.copy()
        mask = np.stack([mask]*3, -1).astype('uint8')

        if img is not None:
            img = mmcv.imread(img)
            img = cv2.resize(img, (MASK_SIZE, MASK_SIZE))
            mask[PAD_SIZE:PAD_SIZE+MASK_SIZE, PAD_SIZE:PAD_SIZE+MASK_SIZE] = img
        cv2.drawContours(mask, [points], -1, (255, 0, 0), 1)

        mask = cv2.rectangle(mask, (PAD_SIZE, PAD_SIZE), (PAD_SIZE +
                             MASK_SIZE, PAD_SIZE+MASK_SIZE), (0, 255, 0), 1)

        mask = cv2.resize(mask, (256, 256))
        cv2.putText(mask, "{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}".format(*self.input.tolist()), (22, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 0, 0), 1)

        if dpi is not None:
            print("Training at size:", self.target.shape)
            plot_images([mask, self.target], mxn=[1, 2], dpi=200)
        return mask


def torch_normalize_input(rbox, im_w, im_h):
    norm_tensor = torch.Tensor([im_w, im_h, im_w, im_h, 90]).reshape([1, 5]).to(rbox.device)
    norm_out = rbox/norm_tensor
    return norm_out

draw_model = IoULossModel()
draw_model.load_state_dict(torch.load(
    "/home/av/gitprojects/yolox/weights/iou_loss_model/last.pth"))

class RIoULoss(nn.Module):
    def __init__(self, img_w, img_h):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.loss_fn = nn.functional.binary_cross_entropy
        self.ip = InputTarget()
        self.step = 0

    def forward(self, pred, target):
        if list(draw_model.parameters())[0].device != pred.device:
            draw_model.to(pred.device)
        draw_model.eval()
        bz = len(pred)
        _pred = torch_normalize_input(pred, self.img_w, self.img_h)
        _target = torch_normalize_input(target, self.img_w, self.img_h)
        mask_pred = draw_model(_pred).reshape(bz, -1)
        mask_target = (draw_model(_target).reshape(bz, -1) > 0.5).float()
        out = self.loss_fn(mask_pred, mask_target, reduction='none').mean(1)
        self.step += 1
        return out


    @torch.no_grad()
    def _debug(self, pred, target):
        pred = torch_normalize_input(pred, self.img_w, self.img_h)
        target = torch_normalize_input(target, self.img_w, self.img_h)

        if list(draw_model.parameters())[0].device != pred.device:
            draw_model.to(pred.device)
        draw_model.eval()
        def get_debug_images(pred):
            reals = []
            for i in range(len(pred)):
                cxcywha = pred[i].cpu().numpy()
                self.ip.set_input(cxcywha)
                mask_pred = self.ip.show()
                input = torch.from_numpy(self.ip.input)[None].cuda()
                approx_pred = draw_model(input)
                approx_pred = nn.functional.interpolate(approx_pred, mask_pred.shape[:2])
                approx_pred = approx_pred.cpu().numpy()[0,0]
                # import ipdb; ipdb.set_trace()
                approx_pred = np.stack([approx_pred]*3, -1)
                pair = np.concatenate([approx_pred, mask_pred], 1)
                pair = np.clip(pair, 0,1)
                reals.append(pair)
            return reals

        vis_preds = get_debug_images(pred)
        vis_targets = get_debug_images(target)
        outs = []
        for vp, vt in zip(vis_preds, vis_targets):
            outs.append(np.concatenate([vp, vt]))
        mmcv.mkdir_or_exist("./cache/debug_riou/")
        plot_images(outs, out_file=f'./cache/debug_riou/{self.step}.jpg')
