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
    def __init__(self, image_path, device='cpu', lmdb_path=None, save_lmdb=False, mode="train"):
        self.image_path = image_path
        self.data = []
        self.lmdb_path = lmdb_path
        self.mode = mode

        # Transform pipeline
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.Normalize(mean=[0.5], std=[0.5]),
            v2.Grayscale(num_output_channels=3)
        ])
        
        if save_lmdb:
            self.save_data()
        else:
            self.load_data()

    def _estimate_lmdb_size(self):
        total_size = 0
        for label in os.listdir(self.image_path):
            label_path = os.path.join(self.image_path, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    total_size += os.path.getsize(img_path)
        
        buffer_factor = 1.5
        return int(total_size * buffer_factor)

    def save_data(self):
        if not os.path.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path)

        lmdb_file = os.path.join(self.lmdb_path, f'{self.mode}_data.lmdb')
        initial_map_size = max(self._estimate_lmdb_size(), int(1e9))
        print(f"Initial Size: {initial_map_size / (1024**3):.2f} GB")

        while True:
            try:
                env = lmdb.open(lmdb_file, map_size=initial_map_size)
                with env.begin(write=True) as txn:
                    index = 0
                    for idx, label in enumerate(os.listdir(self.image_path)):
                        label_path = os.path.join(self.image_path, label)
                        if not os.path.isdir(label_path):
                            continue

                        print(f"Processing {label}...")
                        for img_file in tqdm(os.listdir(label_path), desc=f"Saving {label}"):
                            img_path = os.path.join(label_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                            if img is None:
                                print(f"Skipping {img_path} (invalid image)")
                                continue

                            img = self.transform(img)
                            data = (img, idx)
                            txn.put(f"{index}".encode(), pickle.dumps(data))
                            index += 1
                            self.data.append((img, torch.Tensor([idx])))

                env.close()
                print(f"Saved {self.mode} dataset successfully.")
                break

            except lmdb.MapFullError:
                print(f"LMDB MapFullError: Expanding map_size from {initial_map_size / (1024**3):.2f} GB")
                initial_map_size *= 2

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
