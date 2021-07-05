import albumentations as A
from torchvision.transforms import functional as F


class ToTensor(A.BasicTransform):
    def __init__(self, scaled_input=False, always_apply=True, p=1.0):
        super(ToTensor, self).__init__(always_apply=always_apply, p=p)
        self.scaled_input = scaled_input
    
    @property
    def targets(self):
        return {'image': self.apply}

    def apply(self, img, **params):
        if self.scaled_input:
            img *= 255
        return F.to_tensor(img)


def get_to_tensor():
    return A.Compose(
        [ToTensor()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'iscrowd'])
    )


def get_resize_image(max_size=1333):
    'Faster R-CNN already resizes images during forward https://github.com/pytorch/vision/blob/015378434c1e4be778b2d65eb86f227db20bc8bf/torchvision/models/detection/transform.py#L43'
    return A.Compose(
        [
            A.augmentations.geometric.resize.LongestMaxSize(max_size=max_size),
            ToTensor()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'iscrowd'])
    )


def get_horizontal_flip(p=.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.augmentations.geometric.resize.LongestMaxSize(max_size=1333),
            ToTensor()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'iscrowd'])
    )
