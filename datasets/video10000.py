import scipy
import scipy.misc
import random
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import chainer

class Video10000Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, data_dir, size=256):
        self.base = self.data_loader(path, data_dir)
        self.img_res = (size, size)
        self.crop_ratio = 0.9

    def data_loader(self, path, data_dir):
        base = []
        with open(path, 'r') as f:
            for line in f:
                img_path, label = line.strip('\n').split(' ')
                base.append(['{}/{}'.format(data_dir, img_path), label])
        return base

    def __len__(self):
        return len(self.base)
    
    def transform(self, img_path):
        image = Image.open(img_path)
        h, w = image.size
        crop_size = int(h * self.crop_ratio)
        # Randomly crop a region and flip the image
        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        if random.randint(0, 1):
            image = ImageOps.mirror(image)
        
        bottom = top + crop_size
        right = left + crop_size
        image = image.crop((left, top, right, bottom))
        image = image.resize(self.img_res)
        image = np.asarray(image).astype('float32')
        image = image / 128. - 1.
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        image = image.transpose(2, 0, 1)
        return image

    def get_example(self, i):
        image = self.transform(self.base[i][0])
        label = np.int32(self.base[i][1])
        return image, label