import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transforms import get_to_tensor


PATH_TO_BEESDATASET = os.path.join('..', 'data', 'BeesDataset')


def get_dataset_split(root, number_of_split):
    with open(os.path.join(root, 'splits', '{}.json'.format(number_of_split))) as input_file:
        split = json.load(input_file)
    return split


class BeesDataset(Dataset):
    'Represents BeesDataset for dataloaders.'
    def __init__(self, root, split, transforms=None):
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
        target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        target['image_id'] = image_id
        compute_area = lambda boxes: (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = compute_area(target['boxes'])
        target['iscrowd'] = torch.as_tensor(transformed['iscrowd'], dtype=torch.int64)
        return transformed['image'], target

    def __len__(self):
        return len(self.data)
