from pytorch_lightning import seed_everything
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD, lr_scheduler, Adam
import optuna

from datasets import BeesDataModule
from utils import get_trainer, get_study_storage, get_SGD_objective_fn, get_Adam_objective_fn
from models import ObjectDetector
from transforms import get_horizontal_flip, get_resize_image


SEED = 42


def experiment_1():
    ID = 1
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=1,
        batch_size=1
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_2():
    ID = 2
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=1,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_3():
    ID = 3
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=1,
        batch_size=1
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_4():
    ID = 4
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=1,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_5():
    ID = 5
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=1
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_6():
    ID = 6
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_7():
    ID = 7
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .005, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID,
        auto_lr_find=True
    )
    trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_8():
    ID = 8
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_9():
    ID = 9
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=.9, weight_decay=.0001)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_10():
    ID = 10
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=.9, weight_decay=.0001, nesterov=True)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, .02, optimizer_fn, lr_scheduler_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_11():
    ID = 11
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return Adam(params, lr=lr)
    model = ObjectDetector(model, .001, optimizer_fn)
    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_12():
    ID = 12
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return Adam(params, lr=lr)
    model = ObjectDetector(model, .001, optimizer_fn)
    trainer = get_trainer(
        max_epochs=30,
        min_delta=.001,
        patience=7,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_13():
    ID = 13
    seed_everything(SEED, workers=True)
    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler)
        return model
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(0., 1., False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(.1, .5, False),
        learning_rate=(1e-5, 1e-2, True),
        max_epochs=10,
        limit_train_batches=.3
    )
    study = optuna.create_study(study_name=f'experiment_{ID}', storage=get_study_storage(), direction='maximize', pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_14():
    ID = 14
    seed_everything(SEED, workers=True)
    def get_model(lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = ObjectDetector(model, lr, optimizer_fn, lr_scheduler_fn, update_lr_scheduler)
        return model
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    objective = get_SGD_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        SGD_momentum=(.2, 1., False),
        SGD_weight_decay=(1e-7, 1e-1, True),
        RLROP_factor=(.1, .5, False),
        learning_rate=(1e-4, 1e-2, True),
        max_epochs=10,
        limit_train_batches=.6
    )
    study = optuna.create_study(study_name=f'experiment_{ID}', storage=get_study_storage(), direction='maximize', pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


def experiment_15():
    'Using the best parameters from the experiment_14.'
    ID = 15
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=.8162894810140822, weight_decay=2.866805255260386e-05, nesterov=True)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.31027532484623377, patience=2)
    model = ObjectDetector(model, .005858786517220363, optimizer_fn, lr_scheduler_fn, update_lr_scheduler=True)
    trainer = get_trainer(
        max_epochs=30,
        min_delta=1e-4,
        patience=10,
        version=ID
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


def experiment_16():
    'Using the best parameters from the experiment_14.'
    ID = 16
    seed_everything(SEED, workers=True)
    data_module = BeesDataModule(
        split_id=2,
        batch_size=4,
        transforms={'train': get_resize_image(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return SGD(params, lr=lr, momentum=.8162894810140822, weight_decay=2.866805255260386e-05, nesterov=True)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.31027532484623377, patience=2)
    model = ObjectDetector(model, .005858786517220363, optimizer_fn, lr_scheduler_fn, update_lr_scheduler=True)
    trainer = get_trainer(
        max_epochs=30,
        min_delta=1e-4,
        patience=10,
        version=ID
    )
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
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 2e-3, True),
        beta1=(1e-5, 1., True),
        beta2=(1e-5, 1., True),
        epsilon=(1e-11, 1., True),
        max_epochs=10,
        limit_train_batches=.3
    )
    study = optuna.create_study(study_name=f'experiment_{ID}', storage=get_study_storage(), direction='maximize', pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
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
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 2e-3, True),
        beta1=(.1, 1., False),
        beta2=(.1, 1., False),
        epsilon=(1e-11, 1., True),
        max_epochs=10,
        limit_train_batches=.3
    )
    study = optuna.create_study(study_name=f'experiment_{ID}', storage=get_study_storage(), direction='maximize', pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
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
        transforms={'train': get_horizontal_flip(), 'validate': get_resize_image(), 'test': get_resize_image()}
    )
    objective = get_Adam_objective_fn(
        get_model_fn=get_model,
        data_module=data_module,
        learning_rate=(1e-6, 1e-3, True),
        beta1=(.1, 1., False),
        beta2=(.1, 1., False),
        epsilon=(1e-11, 1., True),
        max_epochs=10,
        limit_train_batches=.6
    )
    study = optuna.create_study(study_name=f'experiment_{ID}', storage=get_study_storage(), direction='maximize', pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
    study.optimize(objective, n_trials=None, timeout=8.8 * 60 * 60)


EXPERIMENTS_DICT = {
    '1': experiment_1,
    '2': experiment_2,
    '3': experiment_3,
    '4': experiment_4,
    '5': experiment_5,
    '6': experiment_6,
    '7': experiment_7,
    '8': experiment_8,
    '9': experiment_9,
    '10': experiment_10,
    '11': experiment_11,
    '12': experiment_12,
    '13': experiment_13,
    '14': experiment_14,
    '15': experiment_15,
    '16': experiment_16,
    '17': experiment_17,
    '18': experiment_18,
    '19': experiment_19,
}
