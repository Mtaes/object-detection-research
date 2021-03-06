from pytorch_lightning import seed_everything
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN,
    ssd300_vgg16,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torch.optim import SGD, lr_scheduler, Adam
import optuna

from datasets import BeesDataModule
from utils import (
    get_trainer,
    get_study_storage,
    get_SGD_objective_fn,
    get_Adam_objective_fn,
    get_transform_objective_fn,
)
from models import ObjectDetector
from transforms import get_horizontal_flip, get_resize_image, get_transforms_test


SEED = 42


def experiment_1():
    ID = 1
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=1, batch_size=1)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_2():
    ID = 2
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=1, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_3():
    ID = 3
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=1, batch_size=1)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_4():
    ID = 4
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=1, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_5():
    ID = 5
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=1)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_6():
    ID = 6
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_7():
    ID = 7
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15, min_delta=0.001, patience=5, version=ID, auto_lr_find=True
    )
    trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_8():
    ID = 8
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_9():
    ID = 9
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=0.9, weight_decay=0.0001)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_10():
    ID = 10
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)

    model = ObjectDetector(model, 0.02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_11():
    ID = 11
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(split_id=2, batch_size=4)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(params, lr=lr)

    model = ObjectDetector(model, 0.001, optimizer_fn)
    trainer = get_trainer(max_epochs=15, min_delta=0.001, patience=5, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_12():
    ID = 12
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(params, lr=lr)

    model = ObjectDetector(model, 0.001, optimizer_fn)
    trainer = get_trainer(max_epochs=30, min_delta=0.001, patience=7, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_13():
    ID = 13
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.0, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-5, 1e-2, True),
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_14():
    ID = 14
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.2, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-4, 1e-2, True),
        max_epochs=10,
        limit_train_batches=0.6,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_15():
    "Using the best parameters from the experiment_14."
    ID = 15
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.8162894810140822,
            weight_decay=2.866805255260386e-05,
            nesterov=True,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31027532484623377, patience=2
        )

    model = ObjectDetector(
        model,
        0.005858786517220363,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
    )
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_16():
    "Using the best parameters from the experiment_14."
    ID = 16
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_resize_image(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.8162894810140822,
            weight_decay=2.866805255260386e-05,
            nesterov=True,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31027532484623377, patience=2
        )

    model = ObjectDetector(
        model,
        0.005858786517220363,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
    )
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_17():
    ID = 17
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn)
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 2e-3, True),
        beta1=(1e-5, 1.0, True),
        beta2=(1e-5, 1.0, True),
        epsilon=(1e-11, 1.0, True),
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_18():
    ID = 18
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn)
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 2e-3, True),
        beta1=(0.1, 1.0, False),
        beta2=(0.1, 1.0, False),
        epsilon=(1e-11, 1.0, True),
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_19():
    ID = 19
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn)
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 1e-3, True),
        beta1=(0.1, 1.0, False),
        beta2=(0.1, 1.0, False),
        epsilon=(1e-11, 1.0, True),
        max_epochs=10,
        limit_train_batches=0.6,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_20():
    "Using the best parameters from the experiment_19."
    ID = 20
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(
            params,
            lr=lr,
            betas=(0.9829382245174512, 0.7877180204367489),
            eps=0.025608314414080815,
        )

    model = ObjectDetector(model, 0.00027172301931541036, optimizer_fn)
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_21():
    "Using the best parameters from the experiment_19."
    ID = 21
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_resize_image(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(
            params,
            lr=lr,
            betas=(0.9829382245174512, 0.7877180204367489),
            eps=0.025608314414080815,
        )

    model = ObjectDetector(model, 0.00027172301931541036, optimizer_fn)
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_22():
    "Using the best parameters from the experiment_18."
    ID = 22
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(
            params,
            lr=lr,
            betas=(0.5341224854215595, 0.14805658519061204),
            eps=0.00012768810784994326,
        )

    model = ObjectDetector(model, 2.2564072901903894e-05, optimizer_fn)
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_23():
    "Use best params from experiment 18 for transform search"
    ID = 23
    seed_everything(SEED, workers=True)

    def get_model_fn():
        lr = 2.2564072901903894e-05

        def optimizer_fn(params, lr):
            return Adam(
                params,
                lr,
                betas=(0.5341224854215595, 0.14805658519061204),
                eps=0.00012768810784994326,
            )

        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn)
        return model

    def get_data_module_fn(train_transforms):
        data_module = BeesDataModule(
            split_id=2,
            batch_size=4,
            transforms={
                "train": train_transforms,
                "validate": get_resize_image(),
                "test": get_resize_image(),
            },
        )
        return data_module

    objective = get_transform_objective_fn(
        get_model_fn=get_model_fn,
        get_data_module_fn=get_data_module_fn,
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_24():
    "Use best params from experiment 19 for transform search"
    ID = 24
    seed_everything(SEED, workers=True)

    def get_model_fn():
        lr = 0.0008392672923136335

        def optimizer_fn(params, lr):
            return Adam(
                params,
                lr,
                betas=(0.8690414117588908, 0.7909366476177144),
                eps=0.05037832040451959,
            )

        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn)
        return model

    def get_data_module_fn(train_transforms):
        data_module = BeesDataModule(
            split_id=2,
            batch_size=4,
            transforms={
                "train": train_transforms,
                "validate": get_resize_image(),
                "test": get_resize_image(),
            },
        )
        return data_module

    objective = get_transform_objective_fn(
        get_model_fn=get_model_fn,
        get_data_module_fn=get_data_module_fn,
        max_epochs=10,
        limit_train_batches=0.6,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_25():
    "Using the best parameters from the experiment_19 and transforms from the experiment_24."
    ID = 25
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_transforms_test(
                blur=True, equalize_trans=True, gauss_noise=True, h_flip=True
            ),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return Adam(
            params,
            lr=lr,
            betas=(0.8690414117588908, 0.7909366476177144),
            eps=0.05037832040451959,
        )

    model = ObjectDetector(model, 0.0008392672923136335, optimizer_fn)
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_26():
    ID = 26
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        backbone = mobilenet_backbone("mobilenet_v2", pretrained=True, fpn=False)
        model = FasterRCNN(
            backbone, num_classes=2, rpn_anchor_generator=anchor_generator
        )
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.0, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-5, 1e-2, True),
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_27():
    ID = 27
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        backbone = mobilenet_backbone("mobilenet_v2", pretrained=True, fpn=False)
        model = FasterRCNN(
            backbone, num_classes=2, rpn_anchor_generator=anchor_generator
        )
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.0, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-5, 1e-2, True),
        max_epochs=10,
        limit_train_batches=0.6,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_28():
    "Using the best parameters from the experiment_27."
    ID = 28
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    backbone = mobilenet_backbone("mobilenet_v2", pretrained=True, fpn=False)
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.5610866722340018,
            weight_decay=1.0038346656017304e-06,
            nesterov=False,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.24684274350020477, patience=2
        )

    model = ObjectDetector(
        model,
        0.009968052280574959,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
    )
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_29():
    ID = 29
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = ssd300_vgg16(num_classes=2)
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.0, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-5, 1e-3, True),
        max_epochs=10,
        limit_train_batches=0.3,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_30():
    ID = 30
    seed_everything(SEED, workers=True)

    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = ssd300_vgg16(num_classes=2)
        model = ObjectDetector(
            model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler
        )
        return model

    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0.0, 1.0, False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(0.1, 0.5, False),
        learning_rate=(1e-5, 1e-3, True),
        max_epochs=10,
        limit_train_batches=0.6,
    )
    study = optuna.create_study(
        study_name=f"experiment_{ID}",
        storage=get_study_storage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=None, timeout=5.5 * 60 * 60)


def experiment_31():
    "Using the best parameters from the experiment_30."
    ID = 31
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = ssd300_vgg16(num_classes=2)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.9010495789282431,
            weight_decay=0.07477941573189864,
            nesterov=False,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31930271676788574, patience=2
        )

    model = ObjectDetector(
        model,
        0.00026381716714924595,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
    )
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=10, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_32():
    "Using the best parameters from the experiment_30."
    ID = 32
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
    )
    model = ssd300_vgg16(num_classes=2, trainable_backbone_layers=3)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.9010495789282431,
            weight_decay=0.07477941573189864,
            nesterov=False,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31930271676788574, patience=2
        )

    model = ObjectDetector(
        model,
        0.00026381716714924595,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
    )
    trainer = get_trainer(max_epochs=30, min_delta=1e-4, patience=6, version=ID)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_33():
    "Using the parameters from the experiment_15."
    ID = 33
    seed_everything(SEED, workers=True)
    train, al_val = [], []
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
        al_data_lists=(train, al_val),
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.8162894810140822,
            weight_decay=2.866805255260386e-05,
            nesterov=True,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31027532484623377, patience=2
        )

    model = ObjectDetector(
        model,
        0.005858786517220363,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
        al_data=(train, al_val),
    )
    trainer = get_trainer(
        max_epochs=30,
        min_delta=1e-4,
        patience=6,
        version=ID,
        al_reload_dataloaders=True,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_34():
    "Using the parameters from the experiment_15."
    ID = 34
    seed_everything(SEED, workers=True)
    train, al_val = [], []
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={
            "train": get_horizontal_flip(),
            "validate": get_resize_image(),
            "test": get_resize_image(),
        },
        al_data_lists=(train, al_val),
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def optimizer_fn(params, lr):
        return SGD(
            params,
            lr=lr,
            momentum=0.8162894810140822,
            weight_decay=2.866805255260386e-05,
            nesterov=True,
        )

    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.31027532484623377, patience=2
        )

    model = ObjectDetector(
        model,
        0.005858786517220363,
        optimizer_fn,
        lr_scheduler_fn,
        update_lr_scheduler=True,
        al_data=(train, al_val),
        al_add=100,
    )
    trainer = get_trainer(
        max_epochs=30,
        min_delta=1e-4,
        patience=6,
        version=ID,
        al_reload_dataloaders=True,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


EXPERIMENTS_DICT = {
    "1": experiment_1,
    "2": experiment_2,
    "3": experiment_3,
    "4": experiment_4,
    "5": experiment_5,
    "6": experiment_6,
    "7": experiment_7,
    "8": experiment_8,
    "9": experiment_9,
    "10": experiment_10,
    "11": experiment_11,
    "12": experiment_12,
    "13": experiment_13,
    "14": experiment_14,
    "15": experiment_15,
    "16": experiment_16,
    "17": experiment_17,
    "18": experiment_18,
    "19": experiment_19,
    "20": experiment_20,
    "21": experiment_21,
    "22": experiment_22,
    "23": experiment_23,
    "24": experiment_24,
    "25": experiment_25,
    "26": experiment_26,
    "27": experiment_27,
    "28": experiment_28,
    "29": experiment_29,
    "30": experiment_30,
    "31": experiment_31,
    "32": experiment_32,
    "33": experiment_33,
    "34": experiment_34,
}
