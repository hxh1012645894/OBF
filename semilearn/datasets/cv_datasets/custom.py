# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Custom dataset for CLIP-based models with folder structure

import os
import gc
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


# CLIP standard normalization
mean, std = {}, {}
mean['custom'] = [0.48145466, 0.4578275, 0.40821073]  # CLIP mean
std['custom'] = [0.26862954, 0.26130258, 0.27577711]  # CLIP std


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_custom(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """
    Get custom dataset for CLIP-based models.

    Supports folder structure:
        data_dir/train/01/*.jpg
        data_dir/train/02/*.jpg
        ...
        data_dir/val/01/*.jpg
        ...

    Args:
        args: arguments
        alg: algorithm name
        name: dataset name (used for subfolder)
        num_labels: number of labeled samples per class
        num_classes: number of classes
        data_dir: data directory
        include_lb_to_ulb: whether to include labeled data in unlabeled set
    """
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    # Use CLIP normalization
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['custom'], std['custom'])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['custom'], std['custom'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['custom'], std['custom'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['custom'], std['custom'])
    ])

    # Data directory structure
    base_dir = os.path.join(data_dir, name.lower()) if name.lower() != 'custom' else data_dir

    # Check for train/val folders
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    if not os.path.exists(train_dir):
        # If no train folder, use base_dir directly
        train_dir = base_dir

    if not os.path.exists(val_dir):
        # If no val folder, use train_dir for evaluation
        val_dir = train_dir

    # Create datasets
    dataset = CustomDataset(root=train_dir, transform=transform_weak, ulb=False, alg=alg)

    # Calculate percentage for labeled data
    total_samples = len(dataset)
    samples_per_class = total_samples // num_classes
    percentage = num_labels / samples_per_class if samples_per_class > 0 else 1.0

    lb_dset = CustomDataset(
        root=train_dir,
        transform=transform_weak,
        ulb=False,
        alg=alg,
        percentage=percentage
    )

    ulb_dset = CustomDataset(
        root=train_dir,
        transform=transform_weak,
        alg=alg,
        ulb=True,
        medium_transform=transform_medium,
        strong_transform=transform_strong,
        include_lb_to_ulb=include_lb_to_ulb,
        lb_index=lb_dset.lb_idx
    )

    eval_dset = CustomDataset(root=val_dir, transform=transform_val, alg=alg, ulb=False)

    return lb_dset, ulb_dset, eval_dset


class CustomDataset(BasicDataset, ImageFolder):
    """
    Custom dataset for CLIP-based models.
    Inherits from BasicDataset and ImageFolder for folder-based data loading.
    """

    def __init__(self, root, transform, ulb, alg, medium_transform=None, strong_transform=None,
                 percentage=-1, include_lb_to_ulb=True, lb_index=None, num_classes=None):
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index if lb_index is not None else {}

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires medium augmentation"

        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat',
                                       'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'],
                                       f"alg {self.alg} requires strong augmentation"

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None

        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)

                if self.percentage != -1:
                    fnames = fnames[:int(len(fnames) * self.percentage)]

                if self.percentage != -1:
                    lb_idx[target_class] = fnames

                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if target_class in self.lb_index and fname in self.lb_index[target_class]:
                            continue

                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

        gc.collect()
        self.lb_idx = lb_idx
        return instances