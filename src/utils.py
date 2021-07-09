import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from coco.coco_utils import convert_to_coco_api
from coco.coco_eval import CocoEvaluator


def get_coco_stats(preds, gt):
    coco_gt = convert_to_coco_api(gt)
    coco_evaluator = CocoEvaluator(coco_gt, ['bbox'])
    for pred in preds:
        coco_evaluator.update(pred)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_stats = coco_evaluator.coco_eval['bbox'].stats
    coco_dict = {'coco_stat_{}'.format(i): stat for i, stat in zip(range(len(coco_stats)), coco_stats)}
    return coco_dict


def get_trainer(max_epochs: int, min_delta: float, patience: int, gpus: int = 1, version=None, auto_lr_find: bool = False, max_time: str = '00:08:40:00'):
    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        logger=CSVLogger(os.path.join('..', 'logs'), 'experiment', version),
        log_every_n_steps=1,
        log_gpu_memory='all',
        max_time=max_time,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor='coco_stat_0', min_delta=min_delta, patience=patience, mode='max'),
            ModelCheckpoint(monitor='coco_stat_0', mode='max')
        ],
        deterministic=True,
        auto_lr_find=auto_lr_find
    )
    return trainer
