from datasets import PATH_TO_BEESDATASET, get_dataset_split, BeesDataset
from transforms import Compose, ToTensor
from utils import get_dataloaders, get_trainer
from models import ObjectDetector
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD, lr_scheduler, Adam
from pytorch_lightning import seed_everything


SEED = 42


def experiment_1():
    ID = 1
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 1)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 1, 'validate': 1, 'test': 1}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_2():
    ID = 2
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 1)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_3():
    ID = 3
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 1)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 1, 'validate': 1, 'test': 1}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_4():
    ID = 4
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 1)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_5():
    ID = 5
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 1, 'validate': 1, 'test': 1}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_6():
    ID = 6
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_7():
    ID = 7
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.tune(model, data_loaders['train'], data_loaders['validate'])
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_8():
    ID = 8
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_9():
    ID = 9
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_10():
    ID = 10
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
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
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


def experiment_11():
    ID = 11
    seed_everything(SEED, workers=True)
    splits = get_dataset_split(PATH_TO_BEESDATASET, 2)
    data_loaders = get_dataloaders(
        DatasetClass=BeesDataset,
        path_to_dataset=PATH_TO_BEESDATASET,
        splits=splits,
        transforms={'train': Compose([ToTensor()]), 'validate': Compose([ToTensor()]), 'test': Compose([ToTensor()])},
        batch_size={'train': 4, 'validate': 4, 'test': 4}
    )

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    def optimizer_fn(params, lr):
        return Adam(params, lr=lr)
    model = ObjectDetector(model, .02, optimizer_fn)

    trainer = get_trainer(
        max_epochs=15,
        min_delta=.001,
        patience=5,
        version=ID
    )
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


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
    '11': experiment_11
}
