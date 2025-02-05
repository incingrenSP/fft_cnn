import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import models
from torchvision.transforms import v2

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
import lmdb
import pickle

from src.FFTConv import *
from src.ImageHandler import *

class AlexNet(nn.Module):
    def __init__(self, input_channels, number_of_classes, apply_fft=False):
        super(AlexNet, self).__init__() 
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        if apply_fft:
            self.convert_layer()

        # Fully connected layers
        self.fc1 = nn.Linear(2304, 4069)
        self.fc2 = nn.Linear(4069, 4069)
        self.fc3 = nn.Linear(4069, number_of_classes)

        # Other
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm = nn.LocalResponseNorm(size=5, k=2)
        self.droput = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU()

    def convert_layer(self):
        self.conv1 = FFTConvNet(self.conv1, 'low')
        self.conv2 = FFTConvNet(self.conv2, 'low')
        self.conv3 = FFTConvNet(self.conv3, 'low')
        self.conv4 = FFTConvNet(self.conv4, 'low')
        self.conv5 = FFTConvNet(self.conv5, 'low')

    def forward(self, x):
        x = self.maxpool(self.norm(self.relu(self.conv1(x))))  # (B, 96, 27, 27)
        x = self.maxpool(self.norm(self.relu(self.conv2(x))))  # (B, 256, 13, 13)
        x = self.relu(self.conv3(x))                           # (B, 384, 13, 13)
        x = self.relu(self.conv4(x))                           # (B, 384, 13, 13)
        x = self.maxpool(self.relu(self.conv5(x)))             # (B, 256, 6, 6)
        x = self.flatten(x)                                    # (B, 9216)
        x = self.droput(self.relu(self.fc1(x)))                # (B, 4096)
        x = self.droput(self.relu(self.fc2(x)))                # (B, 4096)
        x = self.fc3(x)                                 # (B, num_classes)
        return x

def build_data(lmdb_path, REBUILD_DATA):
    if REBUILD_DATA:
        image_path = os.path.join('data', 'train_set')
        train_data = ImageDataset(image_path=image_path, device=device, lmdb_path=lmdb_path, save_lmdb=True)
        
        image_path = os.path.join('data', 'val_set')
        val_data = ImageDataset(image_path=image_path, device=device, lmdb_path=lmdb_path, save_lmdb=True, mode="val")
        REBUILD_DATA = False
    else:
        train_data = ImageDataset(image_path=None, device=device, lmdb_path=lmdb_path, save_lmdb=False)
        val_data = ImageDataset(image_path=None, device=device, lmdb_path=lmdb_path, save_lmdb=False, mode="val")

    return train_data, val_data

def train(model, train_dl, val_dl, loss_fn, optimizer, scheduler, epochs, name, device):
    best_acc = 0.0
    tr_acc_list = []
    val_acc_list = []
    
    # Mixed precision training
    scaler = torch.amp.GradScaler(device)  
    model.train()

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Training loop
        model.train()
        for images, labels in tqdm(train_dl):
            # Move data to GPU
            labels = labels.squeeze().long()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.amp.autocast(device):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Compute metrics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        tr_acc_list.append(epoch_acc.cpu().item())
        print(f"Training Loss: {epoch_loss:0.6f}, Training Accuracy: {epoch_acc:0.6f}")

        # Validation loop
        val_loss, val_acc = validate(model, val_dl, loss_fn, device)
        val_acc_list.append(val_acc.cpu().item())

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            # model.save_model_dict(os.path.join("models", "alex"), f"{name}_model.pth")
            torch.save(model.state_dict(), os.path.join("models", "alex", f"{name}_model.pth"))

        # Step the scheduler
        scheduler.step()

    print('Training Complete.')
    return tr_acc_list, val_acc_list

def validate(model, val_dl, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(val_dl):
            # Move data to GPU
            labels = labels.squeeze().long()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs.data, labels)

            # Compute metrics
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        val_loss = running_loss / total_samples
        val_acc = running_corrects / total_samples
        print(f"Validation Loss: {val_loss:0.6f}, Validation Accuracy: {val_acc:0.6f}")

    return val_loss, val_acc

if __name__ == "__main__":

    IMG_SIZE = 128    # Default: 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    REBUILD_DATA = False
    
    lmdb_path = os.path.join('lmdb')
    train_data, val_data = build_data(lmdb_path, REBUILD_DATA)

    batch_size = 32
    
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size, shuffle=True)
    
    learning_rate = 1e-3
    weight_decay = 1e-6
    
    model = AlexNet(3, 3, False)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    name = "reg_alex"
    tr_acc_list, val_acc_list = train(model, train_dl, val_dl, loss_fn, optimizer, scheduler, epochs=30, name=name, device=device)

    tr_fft_accuracy = np.array(acc_list, dtype=np.float32)
    val_fft_accuracy = np.array(acc_list, dtype=np.float32)
    np.save(os.path.join('models', 'tr_fft_accuracy.npy'), tr_fft_accuracy)
    np.save(os.path.join('models', 'val_fft_accuracy.npy'), val_fft_accuracy)
