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
        min_delta=.0001,
        patience=5,
        version=ID
    )
    trainer.fit(model, data_loaders['train'], data_loaders['validate'])
    trainer.test(test_dataloaders=data_loaders['test'])


EXPERIMENTS_DICT = {
    '1': experiment_1
}
