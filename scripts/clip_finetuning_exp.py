from avcv.all import *
import torch, torch.nn as nn
from dms_drowsiness.video_writer import Board
import onnxruntime
import numpy as np, cv2
from PIL import Image
from ple.all import *
import torch.utils.data as td
import clip
from fastcore.all import *
import pytorch_lightning as pl
CACHE_DF_PATH = '/tmp/eating_cache_data_df.pkl'

# @imemoize
def get_clip_model(device):
    model, processor = clip.load("ViT-B/32", device=device)
    model.requires_grad_(False)
    # model.visual.requires_grad_(True)
    return model, processor

class ClipFinetuningDataset(td.Dataset):
    def __init__(self, img_paths, action):
        self.img_paths = img_paths
        self.action = action

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):


        img_path = self.img_paths[i]

        return preprocess(Image.open(img_path)), self.to_token(self.action[i])[0], self.img_paths[i], self.action[i], action2id[self.action[i]]

    @self_memoize
    def to_token(self, action):
        action_id = action2id[action]
        action = id2action[action_id]
        return clip.tokenize(f"the person is {action}")





class CLIPExp(BaseExp):
    def __init__(self, exp_name='EXPNAME', 
                    batch_size=64, 
                    num_workers=2, 
                    devices=2,
                    strategy='dp', 
                    **kwargs):
        super().__init__()
        store_attr(**kwargs)

    def get_model(self):
        dl = self.get_data_loader().train_dataloader()
        sched = fn_schedule_cosine_with_warmpup_decay_timm(
            num_epochs=self.max_epochs,
            num_steps_per_epoch=len(dl)//self.devices,
            num_epochs_per_cycle=self.max_epochs//self.num_lr_cycles,
            min_lr=1/100,
            cycle_decay=0.7,
        )
        optim = lambda params:torch.optim.Adam(params)
        class ClipLit(LitModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.final_cls = nn.Linear(512, 4)
            def configure_optimizers(self):
                """
                    Setup optimizer and scheduler
                """
                assert self.create_optimizer_fn is not None

                optimizer = self.create_optimizer_fn(self.parameters())
                scheduler = get_scheduler(optimizer, self.create_lr_scheduler_fn)
                return [optimizer], [scheduler]

            def forward_loss(self, batch, batch_idx, mode):
                x,y,paths,actions, y_cls = batch
                with torch.no_grad():
                    encoded_images_logits = self.model.encode_image(x)
                    # encoded_texts_logits = self.model.encode_text(y)

                logits_cls = self.final_cls(encoded_images_logits)
                # encoded_images = nn.functional.normalize(encoded_images_logits)
                # encoded_texts = nn.functional.normalize(encoded_texts_logits)


                # loss_l1 = nn.functional.smooth_l1_loss(encoded_images, encoded_texts)*1000
                # import ipdb; ipdb.set_trace()
                pred_cls = logits_cls.softmax(1).argmax(1)
                acc = (pred_cls==y_cls).float().mean()
                loss_xent = nn.functional.cross_entropy(logits_cls, y_cls)

                loss = loss_xent
                self.log(f"{mode}_loss", loss, prog_bar=True,
                            rank_zero_only=True, on_epoch=True)
                self.log(f"{mode}_acc", acc, prog_bar=True,
                            rank_zero_only=True, on_epoch=True)
                # self.log(f"{mode}_loss_image", loss_image.mean(), prog_bar=True,
                #             rank_zero_only=True, on_epoch=True)
                # self.log(f"{mode}_loss_text", loss_text.mean(), prog_bar=True,
                #             rank_zero_only=True, on_epoch=True)
                return loss

            def training_step(self, b,i): return self.forward_loss(b,i,'training')
            def validation_step(self, b,i): return self.forward_loss(b,i,'val')

        return ClipLit(self.model, create_optimizer_fn=optim,
                                    create_lr_scheduler_fn=sched)

    def get_data_loader(self):
        class PLData(pl.LightningDataModule):
            def __init__(self, **kwargs):
                super().__init__()
                store_attr(**kwargs)

            def train_dataloader(self):
                return td.DataLoader(self.dataset(self.df_train.img_path.tolist(), df_train.action.tolist()), 
                                        self.batch_size, num_workers=self.num_workers,drop_last=True)


            def val_dataloader(self):
                return td.DataLoader(self.dataset(self.df_val.img_path.tolist(), df_val.action.tolist()), 
                                        self.batch_size, num_workers=self.num_workers,drop_last=True)

        return PLData(batch_size=self.batch_size, num_workers=self.num_workers, dataset=ClipFinetuningDataset, df_train=self.df_train, df_val=self.df_val)

    def get_trainer(self, **kwargs):
        from ple.trainer import get_trainer
        return get_trainer(self.exp_name, 
                                max_epochs=self.max_epochs, 
                                gpus=self.devices,
                            strategy='dp' if self.devices==1 else 'ddp',
                            monitor=dict(metric="val_loss", mode="min"),
                            **kwargs)


if __name__ == '__main__':
    df = mmcv.load(CACHE_DF_PATH)
    id2action = {0: 'eating', 1: 'doing nothing', 2: 'using phone', 3: 'smoking'}
    device = "cpu"
    model, preprocess = get_clip_model(device)

    _id2action = dict(enumerate(df['action'].apply(str).unique().tolist()))
    action2id = {v:k for k, v in _id2action.items()}
    id2action = {k:v for k, v in _id2action.items()}
    logger.info(f'{action2id=}')


    df_shufle = df.sample(frac=1)
    n = int(.2*len(df_shufle))
    df_train = df_shufle[n:]
    df_val = df_shufle[:n]
    clip_model, processor = get_clip_model('cpu')

    # if is_interactive():
    # num_workers = 0
    # devices = 1
    # overfit_batches=10
    # else:
    num_workers = 4
    devices = 8
    overfit_batches=None

    clip_exp = CLIPExp(exp_name=f'finetune_clip/', batch_size=128, 
                    num_workers=num_workers, devices=devices, model=clip_model, max_epochs=10, df_train=df_train, df_val=df_val)
    clip_lit_model = clip_exp.get_model()
    clip_trainer = clip_exp.get_trainer()
    clip_data = clip_exp.get_data_loader()
    clip_trainer.fit(clip_lit_model, clip_data, )