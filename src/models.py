from time import time

from pytorch_lightning import LightningModule

from utils import get_coco_stats


class ObjectDetector(LightningModule):
    def __init__(self, model, learning_rate, optimizer_fn, lr_scheduler_fn=None, update_lr_scheduler: bool = False):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_fn = optimizer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.update_lr_scheduler = update_lr_scheduler

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
    
    def validation_step(self, batch, batch_idx):
        return self._val_test_step(batch)

    def validation_epoch_end(self, validation_step_outputs):
        coco_dict = self._coco_eval(validation_step_outputs)
        self.log('phase', 1)
        self.log_dict(coco_dict)
    
    def test_step(self, batch, batch_idx):
        return self._val_test_step(batch)

    def test_epoch_end(self, test_step_outputs):
        coco_dict = self._coco_eval(test_step_outputs)
        self.log('phase', 2)
        self.log_dict(coco_dict)
