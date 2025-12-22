import os
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTEC(Dataset):
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    def __init__(
            self,
            root: str,
            class_name: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            **kwargs):
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]

        if self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
            self.class_name = None
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()

        self.transform = transform
        if transform is None or transform == 'None':
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.LANCZOS),
                T.CenterCrop(kwargs.get('crp_size')),

                T.ToTensor(),
                T.Normalize(kwargs.get('norm_mean'), kwargs.get('norm_std'))])

        self.target_transform = target_transform
        if target_transform is None or target_transform == 'None':
            self.target_transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.NEAREST),
                T.CenterCrop(kwargs.get('crp_size')),
                T.ToTensor()])

        self.token_transform = T.Compose([
            T.Resize(224, Image.LANCZOS),
            T.CenterCrop(224),
            T.ToTensor()])

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

    def __getitem__(self, idx: int):

        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], \
        self.img_types[idx]

        if self.class_name is None:
            class_name = image_path.split('/')[-4]
        else:
            class_name = self.class_name

        image = Image.open(image_path)
        if class_name in ['zipper', 'screw', 'grid']:
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        token_image = self.token_transform(image)
        image = self.transform(image)

        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.target_transform(mask)
        return image, token_image, label, mask, image_path, img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        image_dir = os.path.join(self.root, self.class_name, phase)
        mask_dir = os.path.join(self.root, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        return image_paths, labels, mask_paths, types

    def _load_all_data(self):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []
        for class_name in self.CLASS_NAMES:
            self.class_name = class_name
            image_paths, labels, mask_paths, types = self._load_data()
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_types.extend(types)
        return all_image_paths, all_labels, all_mask_paths, all_types


    def _check_exists(self) -> bool:
        if not os.path.exists(self.root):
            return False
        return True
