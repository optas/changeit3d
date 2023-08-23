"""
Basic functions to deal with image-based I/O.
Originally created sometime around 2019, for Python 3.x
2022 Panos Achlioptas (https://optas.github.io)
"""

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]


class ImageClassificationDataset(Dataset):
    def __init__(self, image_files, labels=None, img_transform=None, rgb_only=True):
        super(ImageClassificationDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])

        if self.rgb_only and img.mode != 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        label = []
        if self.labels is not None:
            label = self.labels[index]

        res = {'image': img, 'label': label, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)


def image_transformation(img_dim, lanczos=True):
    """simple transformation/pre-processing of image data.
    """

    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    img_transforms = dict()
    img_transforms['train'] = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])

    # Use same transformations as in train (since no data-augmentation is applied in train)
    img_transforms['test'] = img_transforms['train']
    img_transforms['val'] = img_transforms['train']
    img_transforms['rest'] = img_transforms['train']
    return img_transforms

