from time import time

import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from utils import get_coco_stats


class ObjectDetector(LightningModule):
    def __init__(self, model, learning_rate, optimizer_fn, lr_scheduler_fn=None, update_lr_scheduler: bool = False, al_data=None, al_add: int = 200):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_fn = optimizer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.update_lr_scheduler = update_lr_scheduler
        self.al_data = al_data
        self.al_add = al_add

    def forward(self, x):
        preds = self.model(x)
        return preds

    def training_step(self, batch, batch_idx):
        self.log('phase', 0)
        self.log('step_start', time())
        x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss_dict.values())
        loss_dict = {key: item.detach() for key, item in loss_dict.items()}
        loss_dict['loss'] = loss
        self.log('step_end', time())
        self.log_dict(loss_dict)
        return loss_dict

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), self.learning_rate)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = self.lr_scheduler_fn(optimizer)
            if self.update_lr_scheduler:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': lr_scheduler,
                        'monitor': 'coco_stat_0'
                        }
                    }
            else:
                return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer
    
    def _val_test_step(self, batch):
        'Validation and test steps in this case are the same.'
        x, y = batch
        pred = self(x)
        res = {target['image_id'].item(): output for target, output in zip(y, pred)}
        img_meta = [{'height': img.shape[-2], 'width':img.shape[-1]} for img in x]
        return res, tuple(zip(img_meta, y))
    
    def _coco_eval(self, outputs):
        preds, gt = zip(*outputs)
        # TODO Find better solution
        tmp_gt = []
        for x in gt:
            tmp_gt.extend(x)
        coco_dict = get_coco_stats(preds, tmp_gt)
        return coco_dict
    
    def _al_val_step(self, batch):
        x, y = batch
        base_preds = self(x)
        ious = []
        for n in range(10):
            new_x = [img + (torch.randn(img.size(), device=img.device) * n / 10) for img in x]
            preds = self(new_x)
            ious.append([torchvision.ops.box_iou(p['boxes'], gt['boxes']) for p, gt in zip(preds, base_preds)])
        ious_mean = []
        for i in range(len(x)):
            ious_sum = None
            for iou in ious:
                if ious_sum is None:
                    ious_sum = iou[i]
                else:
                    shape_pad_sum = []
                    shape_pad = []
                    for dim in range(2):
                        shape_pad.append(max(ious_sum.shape[dim] - iou[i].shape[dim], 0))
                        shape_pad_sum.append(max(iou[i].shape[dim] - ious_sum.shape[dim], 0))
                    ious_sum = F.pad(ious_sum, (0, shape_pad_sum[1], 0, shape_pad_sum[0]))
                    tmp_iou = F.pad(iou[i], (0, shape_pad[1], 0, shape_pad[0]))
                    ious_sum += tmp_iou
            ious_mean.append(ious_sum / len(ious))
        ious_mean_sum = [(y['image_id'].item(), torch.sum(ious_mean_img).item()) for y, ious_mean_img in zip(y, ious_mean)]
        return ious_mean_sum
    
    def _al_val(self, validation_step_outputs):
        unpacked_outputs = []
        for step in validation_step_outputs:
            unpacked_outputs.extend(step)
        unpacked_outputs.sort(key=lambda x: x[1])
        for output in unpacked_outputs[:self.al_add]:
            self.al_data[0].append(self.al_data[1][output[0]])
        to_remove = unpacked_outputs[:self.al_add]
        to_remove.sort(key=lambda x: x[0], reverse=True)
        for output in to_remove:
            self.al_data[1].pop(output[0])
    
    def validation_step(self, batch, batch_idx,  dataloader_idx=None):
        if dataloader_idx == 1:
            return self._al_val_step(batch)
        else:
            return self._val_test_step(batch)

    def validation_epoch_end(self, validation_step_outputs):
        if self.al_data is not None:
            self._al_val(validation_step_outputs[1])
            validation_step_outputs = validation_step_outputs[0]
        coco_dict = self._coco_eval(validation_step_outputs)
        self.log('phase', 1)
        self.log_dict(coco_dict)
    
    def test_step(self, batch, batch_idx):
        return self._val_test_step(batch)

    def test_epoch_end(self, test_step_outputs):
        coco_dict = self._coco_eval(test_step_outputs)
        self.log('phase', 2)
        self.log_dict(coco_dict)
