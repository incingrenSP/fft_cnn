import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import os
import cv2
import lmdb
import pickle
from tqdm import tqdm

IMG_SIZE = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageDataset(Dataset):
    def __init__(self, image_path, device='cpu', lmdb_path=None, save_lmdb=False, map_size=None, mode="train"):
        self.image_path = image_path
        self.data = []
        self.lmdb_path = lmdb_path
        self.mode = mode
        
        # Augmentation pipeline
        self.augmentation = v2.Compose([
            v2.ToImage(),
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # v2.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5,], std=[0.5,]),
            v2.Grayscale(num_output_channels=3)
        ])
        
        # Transform pipeline
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.Normalize(mean=[0.5,], std=[0.5,]),
            v2.Grayscale(num_output_channels=3)
        ])
        
        if save_lmdb:
            self.save_data()
        else:
            self.load_data()

    def save_data(self):
        lmdb_file = os.path.join(self.lmdb_path, f'{self.mode}_data.lmdb')
        env = lmdb.open(lmdb_file, map_size=int(3e9))
        with env.begin(write=True) as txn:
            index = 0
            for idx, label in enumerate(os.listdir(os.path.join(self.image_path))):
                label_path = os.path.join(self.image_path, label)
                print(os.path.join(self.image_path, label))
                for img_file in tqdm(os.listdir(label_path), desc=f"Saving {label}"):
                    img = cv2.imread(os.path.join(label_path, img_file), cv2.IMREAD_GRAYSCALE)
                    if self.mode == "train":
                        img = self.augmentation(img)
                    elif self.mode == "val" or self.mode == "test":
                        img = self.transform(img)
                    
                    data = (img, idx)
                    txn.put(f"{index}".encode(), pickle.dumps(data))
                    index += 1
                    self.data.append((img, torch.Tensor([idx])))
        env.close()
        print(f"Saved {self.mode} dataset")

    def load_data(self):
        lmdb_file = os.path.join(self.lmdb_path, f'{self.mode}_data.lmdb')

        if not os.path.exists(lmdb_file):
            raise FileNotFoundError(f"LMDB file {lmdb_file} not found")

        env = lmdb.open(lmdb_file, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                loaded_data = pickle.loads(value)    
                img, label = loaded_data
                self.data.append((img, torch.Tensor([label])))

        env.close()
        print(f"Loaded {self.mode} dataset")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]