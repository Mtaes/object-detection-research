from datasets import PATH_TO_BEESDATASET, get_dataset_split, BeesDataset
from transforms import Compose, ToTensor
from utils import get_dataloaders, get_trainer
from models import ObjectDetector
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD, lr_scheduler
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
    def optimizer_fn(params):
        return SGD(params, lr=.005)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, optimizer_fn, lr_scheduler_fn)

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
    def optimizer_fn(params):
        return SGD(params, lr=.005)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, optimizer_fn, lr_scheduler_fn)

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
    def optimizer_fn(params):
        return SGD(params, lr=.02)
    def lr_scheduler_fn(optimizer):
        return lr_scheduler.StepLR(optimizer, step_size=3)
    model = ObjectDetector(model, optimizer_fn, lr_scheduler_fn)

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
    '3': experiment_3
}
