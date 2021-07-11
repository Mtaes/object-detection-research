import os
from typing import Optional, Union
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback

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


def get_trainer(max_epochs: int, min_delta: Optional[float] = None, patience: Optional[int] = None, version=None, auto_lr_find: bool = False,
                max_time: str = '00:08:40:00', trial: Optional[Trial] = None, limit_train_batches: Union[int, float] = 1.):
    if trial is None:
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor='coco_stat_0', mode='max')
        ]
        if min_delta is not None and patience is not None:
            callbacks.append(EarlyStopping(monitor='coco_stat_0', min_delta=min_delta, patience=patience, mode='max'))
    else:
        callbacks = [PyTorchLightningPruningCallback(trial, monitor='coco_stat_0')]
    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        logger=CSVLogger(os.path.join('..', 'logs'), 'experiment', version) if trial is None else False,
        log_every_n_steps=1,
        log_gpu_memory='all',
        max_time=max_time,
        callbacks=callbacks,
        deterministic=True,
        auto_lr_find=auto_lr_find,
        checkpoint_callback=True if trial is None else False,
        terminate_on_nan=True,
        num_sanity_val_steps=0,
        limit_train_batches=limit_train_batches
    )
    return trainer


def get_study_storage():
    storage_dir = os.path.join('..', 'logs', 'experiment')
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return 'sqlite:///{}'.format(os.path.join(storage_dir, 'studies.db'))
