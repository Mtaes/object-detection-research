import os
import json
from typing import Optional
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from albumentations import Compose

from transforms import get_to_tensor
from coco.utils import collate_fn


PATH_TO_BEESDATASET = os.path.join('..', 'data', 'BeesDataset')


def get_dataset_split(root, split_id) -> dict:
    with open(os.path.join(root, 'splits', '{}.json'.format(split_id))) as input_file:
        split = json.load(input_file)
    return split


class BeesDataset(Dataset):
    'Represents BeesDataset for dataloaders.'
    def __init__(self, root: str, split: list, transforms: Optional[Compose] = None):
        self.root = root
        self.transforms = get_to_tensor() if transforms is None else transforms
        with open(os.path.join(root, 'boxes.json'), 'r') as input_file:
            self.data = json.load(input_file)
        self.data = list(filter(lambda e: len(e['boxes']) > 0 and e['image_name'] in split, self.data))

    def __getitem__(self, idx):
        image_data = self.data[idx]
        file_name = image_data['image_name']
        boxes = []
        for box in image_data['boxes']:
            boxes.append([
                box['xmin'],
                box['ymin'],
                box['xmin'] + box['width'],
                box['ymin'] + box['height']
            ])
        labels = np.ones((len(boxes),))
        image_id = torch.tensor([idx])
        iscrowd = np.zeros((len(boxes),))
        img_path = os.path.join(self.root, 'images', file_name)
        img = Image.open(img_path).convert('RGB')
        transformed = self.transforms(image=np.asarray(img), bboxes=boxes, labels=labels, iscrowd=iscrowd)
        target = {}
        target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes']) > 0 else torch.zeros((0,4), dtype=torch.float32)
        target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        target['image_id'] = image_id
        target['area'] = (lambda boxes: (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))(target['boxes'])
        target['iscrowd'] = torch.as_tensor(transformed['iscrowd'], dtype=torch.int64)
        return transformed['image'], target

    def __len__(self):
        return len(self.data)


class BeesDataModule(LightningDataModule):
    def __init__(self, split_id, batch_size: int, transforms: Optional[dict] = None, num_workers: Optional[int] = None, data_dir: str = PATH_TO_BEESDATASET):
        super().__init__()
        self.split_id = split_id
        self.batch_size = batch_size
        self.transforms = {} if transforms is None else transforms
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.data_dir = data_dir
        self.splits = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
    
    def setup(self, stage: Optional[str]):
        if self.splits is None:
            self.splits = get_dataset_split(self.data_dir, self.split_id)
        if stage in (None, 'fit'):
            self.data_train = BeesDataset(self.data_dir, self.splits['train'], self.transforms.get('train'))
        if stage in (None, 'validate', 'fit'):
            self.data_val = BeesDataset(self.data_dir, self.splits['validate'], self.transforms.get('validate'))
        if stage in (None, 'test'):
            self.data_test = BeesDataset(self.data_dir, self.splits['test'], self.transforms.get('test'))
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True)
