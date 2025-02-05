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

from src.ImageHandler import *
from src.FFTConv import *

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
    
    # For mixed precision training
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
            # print("Data loaded")
            # Move data to GPU
            labels = labels.squeeze().long()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.amp.autocast(device):
                outputs = model(images)
                loss = loss_fn(outputs[0], labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Compute metrics
            _, preds = torch.max(outputs[0], 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        tr_acc_list.append(epoch_acc.cpu().item())
        print(f"Training Loss: {epoch_loss:0.6f}, Training Accuracy: {epoch_acc:0.6f}")

        # Validation loop
        val_loss, val_acc = validate(model, val_dl, loss_fn)
        val_acc_list.append(val_acc.cpu().item())

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_state_dict(os.path.join("models", "google"), f"{name}_model.pth"))
            # torch.save(model.state_dict(), os.path.join("models", "google", f"{name}_model.pth"))

        # Step the scheduler
        scheduler.step()

    print('Training Complete.')
    return tr_acc_list, val_acc_list

def validate(model, val_dl, loss_fn):
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
            # print("Output shape: ", outputs.shape)
            loss = loss_fn(outputs, labels)

            # Compute metrics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        val_loss = running_loss / total_samples
        val_acc = running_corrects / total_samples
        print(f"Validation Loss: {val_loss:0.6f}, Validation Accuracy: {val_acc:0.6f}")

    return val_loss, val_acc

if __name__ == "__main__":
    IMG_SIZE = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    REBUILD_DATA = False

    lmdb_path = os.path.join('lmdb')
    train_data, val_data = build_data(lmdb_path, REBUILD_DATA)

    batch_size = 32
    
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size, shuffle=True)

    # Load GoogleNet
    learning_rate = 1e-2
    weight_decay = 1e-4
    
    model = FFTGoogle(apply_fft=False, device=device)
    # model = FFTGoogle(apply_fft=True, device=device)

    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    name = "reg_google"
    tr_acc_list, val_acc_list = train(model, train_dl, val_dl, loss_fn, optimizer, scheduler, epochs=30, name, device)

    tr_reg_accuracy = np.array(tr_acc_list, dtype=np.float32)
    val_reg_accuracy = np.array(val_acc_list, dtype=np.float32)
    np.save(os.path.join('models', 'tr_reg_accuracy.npy'), tr_reg_accuracy)
    np.save(os.path.join('models', 'val_reg_accuracy.npy'), val_reg_accuracy)