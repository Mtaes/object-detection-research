import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


PATH_TO_BEESDATASET = os.path.join('..', 'data', 'BeesDataset')


def get_dataset_split(root, number_of_split):
    with open(os.path.join(root, 'splits', '{}.json'.format(number_of_split))) as input_file:
        split = json.load(input_file)
    return split


class BeesDataset(Dataset):
    'Represents BeesDataset for dataloaders.'
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.transforms = transforms
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
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        img_path = os.path.join(self.root, 'images', file_name)
        img = Image.open(img_path).convert('RGB')
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.data)
