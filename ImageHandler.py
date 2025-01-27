import os
import cv2
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset

IMG_SIZE = 128

class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.data = []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.Normalize(mean=[0.5,], std=[0.5,]),
            v2.Grayscale(num_output_channels=3)
        ])
        self.load_data()

    def load_data(self):
        for idx, label in enumerate(os.listdir(os.path.join(self.image_path))):
            print(os.path.join(self.image_path, label))
            for img_file in tqdm(os.listdir(os.path.join(self.image_path, label))):
                img = cv2.imread(os.path.join(self.image_path, label, img_file), cv2.IMREAD_GRAYSCALE)

                img = self.transform(img)

                idx = torch.Tensor([idx])
                self.data.append((img, idx))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]