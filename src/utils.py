import os
from typing import Optional, Union
from pathlib import Path

import torch
from torch.optim import SGD, lr_scheduler, Adam
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback

from coco.coco_utils import convert_to_coco_api
from coco.coco_eval import CocoEvaluator
from transforms import get_transforms_test


def get_coco_stats(preds, gt):
    coco_gt = convert_to_coco_api(gt)
    coco_evaluator = CocoEvaluator(coco_gt, ["bbox"])
    for pred in preds:
        coco_evaluator.update(pred)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_stats = coco_evaluator.coco_eval["bbox"].stats
    coco_dict = {
        "coco_stat_{}".format(i): stat
        for i, stat in zip(range(len(coco_stats)), coco_stats)
    }
    return coco_dict


def get_trainer(
    max_epochs: int,
    min_delta: Optional[float] = None,
    patience: Optional[int] = None,
    version=None,
    auto_lr_find: bool = False,
    max_time: str = "00:08:40:00",
    trial: Optional[Trial] = None,
    limit_train_batches: Union[int, float] = 1.0,
    al_reload_dataloaders: bool = False,
):
    if trial is None:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="coco_stat_0", mode="max"),
        ]
        if min_delta is not None and patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="coco_stat_0",
                    min_delta=min_delta,
                    patience=patience,
                    mode="max",
                )
            )
    else:
        callbacks = [PyTorchLightningPruningCallback(trial, monitor="coco_stat_0")]
    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        logger=CSVLogger(os.path.join("..", "logs"), "experiment", version)
        if trial is None
        else False,
        log_every_n_steps=1,
        log_gpu_memory="all",
        max_time=max_time,
        callbacks=callbacks,
        deterministic=True,
        auto_lr_find=auto_lr_find,
        checkpoint_callback=True if trial is None else False,
        terminate_on_nan=True,
        num_sanity_val_steps=0,
        limit_train_batches=limit_train_batches,
        reload_dataloaders_every_n_epochs=1 if al_reload_dataloaders else 0,
    )
    return trainer


def get_study_storage():
    storage_dir = os.path.join("..", "logs", "experiment")
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return "sqlite:///{}".format(os.path.join(storage_dir, "studies.db"))


def get_SGD_objective_fn(
    get_model_fn,
    data_module: LightningDataModule,
    SGD_momentum: Union[float, tuple],
    SGD_weight_decay: Union[float, tuple],
    RLROP_factor: Union[float, tuple],
    learning_rate: Union[float, tuple],
    max_epochs: int,
    limit_train_batches: Union[int, float],
    SGD_nesterov: Optional[bool] = None,
):
    def objective(trial: Trial):
        def suggest_float_wrapper(name: str, args: Union[float, tuple]):
            return (
                args
                if isinstance(args, float)
                else trial.suggest_float(
                    name=name, low=args[0], high=args[1], log=args[2]
                )
            )

        momentum = suggest_float_wrapper("SGD_momentum", SGD_momentum)
        weight_decay = suggest_float_wrapper("SGD_weight_decay", SGD_weight_decay)
        nesterov = (
            SGD_nesterov
            if isinstance(SGD_nesterov, bool)
            else trial.suggest_categorical("SGD_nesterov", [True, False])
        )

        def optimizer_fn(params, lr):
            return SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )

        factor = suggest_float_wrapper("ReduceLROnPlateau_factor", RLROP_factor)

        def lr_scheduler_fn(optimizer):
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=factor, patience=2
            )

        lr = suggest_float_wrapper("learning_rate", learning_rate)
        model = get_model_fn(
            lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler=True
        )
        trainer = get_trainer(
            max_epochs=max_epochs, trial=trial, limit_train_batches=limit_train_batches
        )
        try:
            trainer.fit(model, datamodule=data_module)
        except ValueError as err:
            print(err)
            return None
        else:
            return trainer.callback_metrics["coco_stat_0"].item()

    return objective


def get_Adam_objective_fn(
    get_model_fn,
    data_module: LightningDataModule,
    learning_rate: tuple,
    beta1: tuple,
    beta2: tuple,
    epsilon: tuple,
    max_epochs: int,
    limit_train_batches: Union[int, float],
):
    def objective(trial: Trial):
        betas = (
            trial.suggest_float(
                name="beta1", low=beta1[0], high=beta1[1], log=beta1[2]
            ),
            trial.suggest_float(
                name="beta2", low=beta2[0], high=beta2[1], log=beta2[2]
            ),
        )
        eps = trial.suggest_float(
            name="epsilon", low=epsilon[0], high=epsilon[1], log=epsilon[2]
        )

        def optimizer_fn(params, lr):
            return Adam(params, lr, betas, eps)

        lr = trial.suggest_float(
            name="learning_rate",
            low=learning_rate[0],
            high=learning_rate[1],
            log=learning_rate[2],
        )
        model = get_model_fn(lr, optimizer_fn)
        trainer = get_trainer(
            max_epochs=max_epochs, trial=trial, limit_train_batches=limit_train_batches
        )
        try:
            trainer.fit(model, datamodule=data_module)
        except ValueError as err:
            print(err)
            return None
        else:
            return trainer.callback_metrics["coco_stat_0"].item()

    return objective


def get_transform_objective_fn(
    get_model_fn, get_data_module_fn, max_epochs, limit_train_batches
):
    def objective(trial: Trial):
        train_transforms = get_transforms_test(
            h_flip=trial.suggest_categorical("h_flip", (True, False)),
            rotate=trial.suggest_categorical("rotate", (True, False)),
            random_crop=trial.suggest_categorical("random_crop", (True, False)),
            blur=trial.suggest_categorical("blur", (True, False)),
            gauss_noise=trial.suggest_categorical("gauss_noise", (True, False)),
            equalize_trans=trial.suggest_categorical("equalize", (True, False)),
        )
        data_module = get_data_module_fn(train_transforms)
        model = get_model_fn()
        trainer = get_trainer(
            max_epochs=max_epochs, trial=trial, limit_train_batches=limit_train_batches
        )
        try:
            trainer.fit(model, datamodule=data_module)
        except ValueError as err:
            print(err)
            return None
        else:
            return trainer.callback_metrics["coco_stat_0"].item()

    return objective
