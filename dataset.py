import pickle
import re

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from config import DATA_MODES, LABEL_ENCODER_NAME, IMAGE_SIZE
import cv2
from torchvision import transforms


class IDAODataset(Dataset):
    def __init__(self, files, mode, augmentation=None):
        self.files = files
        self.mode = mode
        self.augmentation = augmentation
        if self.mode not in DATA_MODES:
            raise NameError('Current mode not in data modes')
        self.len = len(self.files)
        if mode == 'val':
            self.label_encoder = pickle.load(open(LABEL_ENCODER_NAME, 'rb'))
        else:
            self.label_encoder = LabelEncoder()

        if mode != 'test':
            self.labels = {'classification': [path.parent.name for path in self.files],
                           'regression': [int(re.search(r'[EN]R_(\d+)', file.name)
                                              .group(1))
                                          for file in self.files]}
        if self.mode == 'train':
            self.label_encoder.fit(self.labels['classification'])
            with open('label_encoder.pkl', 'wb') as le_dump:
                pickle.dump(self.label_encoder, le_dump)

    def __len__(self):
        return self.len

    @staticmethod
    def crop(img, size=IMAGE_SIZE, stride=32):
        m, ms = (0, 0), 0
        for i in range(0, img.shape[0] - size[0], stride):
            for j in range(0, img.shape[1] - size[1], stride):
                cs = img[
                     i: i + size[0],
                     j: j + size[1]
                     ].sum()
                if cs > ms:
                    m, ms = (i, j), cs
        return img[m[0]: m[0] + size[0], m[1]: m[1] + size[1]]

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def _prepare_sample(self, image):
        """Here image could be processed"""
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self.crop(np.array(image))

    def show_image(self, index):
        image = self.load_sample(self.files[index])
        image.show()

    def __getitem__(self, index):
        if index < 0 or index > len(self.files):
            raise IndexError()
        image = self.load_sample(self.files[index])
        image = image.resize(IMAGE_SIZE)
        image = np.array(np.array(image) / 225, dtype='float32')
        image = self._prepare_sample(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45], std=[0.246])  # random_values, should be found
        ])

        image = transform(image)
        if self.mode == 'test':
            return image
        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]
        label_cl = self.label_encoder.transform([self.labels['classification'][index]]).item()
        label_reg = self.labels['regression'][index]
        return image, label_cl, label_reg
