import os
import numpy as np
from scipy.ndimage import imread

class DSDataLoader:

    def __init__(self, path='/home/leesy714/dataset/ds', batch_size=128, normalize=None):
        self.path = path
        self.batch_size = batch_size
        if normalize is None:
            self.mean = np.zeros(3)
            self.std = np.ones(3)
        else:
            self.mean = normalize[0]
            self.std = normalize[1]

        filenames = np.array(os.listdir(self.path))
        self.length = len(filenames) / self.batch_size

    def __len__(self):
        return self.length

    def load_batch(self, shuffle=False):
        filenames = np.array(os.listdir(self.path))
        n = len(filenames)
        if shuffle:
            idx = np.random.permutation(n)
            filenames = filenames[idx]
        for i in range(0, n, self.batch_size):
            X = []
            for b in range(i, i + self.batch_size):
                filepath = os.path.join(self.path, filenames[b])
                img = imread(filepath, mode='RGB')
                img = img / 255.0
                img = (img - self.mean) / self.std
                X.append(img)
            X = np.array(X).transpose((0,3,1,2))
            yield X
        
        
