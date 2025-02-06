import torch
import torchvision
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

import os
import cv2
import numpy as np
import lmdb
import pickle

from src.FFTConv import *
from src.ImageHandler import *

def get_predictions(model, test_dl, device):
    """Get model predictions on test dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dl):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # List of model predictions
            all_preds.extend(preds.cpu().numpy())
            # List of ground truths
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_metrics(precision1, recall1, f1_1, precision2, recall2, f1_2, class_names):
    """Bar Graph Plot for Precision, Recall and F1-Score"""
    x = np.arange(len(class_names))
    width = 0.3

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    # Precision Bar Graph
    ax[0].bar(x - width/2, precision1, width, label="Model 1", color="blue")
    ax[0].bar(x + width/2, precision2, width, label="Model 2", color="orange")
    ax[0].set_title("Precision Comparison")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(class_names)

    # Recall Bar Graph
    ax[1].bar(x - width/2, recall1, width, color="blue")
    ax[1].bar(x + width/2, recall2, width, color="orange")
    ax[1].set_title("Recall Comparison")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(class_names)

    # F1-score Bar Graph
    ax[2].bar(x - width/2, f1_1, width, color="blue")
    ax[2].bar(x + width/2, f1_2, width, color="orange")
    ax[2].set_title("F1-score Comparison")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(class_names)

    for a in ax:
        a.legend(["FFT", "Regular"])
        a.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def compute_metrics(ground_truths, predictions):
    """Calculate Precision, Recall and F1-Score"""
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average=None)
    return precision, recall, f1

def plot_confusion_matrix(y_true, y_pred1, y_pred2, class_names):
    cm1 = confusion_matrix(y_true, y_pred1)
    cm2 = confusion_matrix(y_true, y_pred2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title("Confusion Matrix - FFT")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")

    sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title("Confusion Matrix - Regular")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

def plot_accuracy_graphs(
    model1_train_acc, model1_val_acc,
    model2_train_acc, model2_val_acc,
    num_epochs
):
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # FFT Training vs Validation
    ax[0, 0].plot(epochs, model1_train_acc, label="Train", marker="o", color="blue")
    ax[0, 0].plot(epochs, model1_val_acc, label="Validation", marker="s", color="red")
    ax[0, 0].set_title("FFT: Training vs Validation Accuracy")
    ax[0, 0].set_xlabel("Epochs")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].legend()

    # Regular Training vs Validation
    ax[0, 1].plot(epochs, model2_train_acc, label="Train", marker="o", color="blue")
    ax[0, 1].plot(epochs, model2_val_acc, label="Validation", marker="s", color="red")
    ax[0, 1].set_title("Regular: Training vs Validation Accuracy")
    ax[0, 1].set_xlabel("Epochs")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()

    # FFT Validation vs Regular Validation
    ax[1, 0].plot(epochs, model1_val_acc, label="FFT", marker="o", color="blue")
    ax[1, 0].plot(epochs, model2_val_acc, label="Regular", marker="s", color="orange")
    ax[1, 0].set_title("Validation Accuracy: FFT vs Regular")
    ax[1, 0].set_xlabel("Epochs")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].legend()

    # FFT Training vs Regular Training
    ax[1, 1].plot(epochs, model1_train_acc, label="FFT", marker="o", color="blue")
    ax[1, 1].plot(epochs, model2_train_acc, label="Regular", marker="s", color="orange")
    ax[1, 1].set_title("Training Accuracy: FFT vs Regular")
    ax[1, 1].set_xlabel("Epochs")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()

def compare_models(model1, model2, test_dl, class_names, 
                   model1_train_acc, model1_val_acc,
                   model2_train_acc, model2_val_acc,
                   num_epochs, device):
    """Base function for comparing fft and base models"""
    y_true, y_pred1 = get_predictions(model1, test_dl, device)
    _, y_pred2 = get_predictions(model2, test_dl, device)

    precision1, recall1, f1_1 = compute_metrics(y_true, y_pred1)
    precision2, recall2, f1_2 = compute_metrics(y_true, y_pred2)

    print("\nFFT Metrics:\n", classification_report(y_true, y_pred1, target_names=class_names))
    print("\nRegular Metrics:\n", classification_report(y_true, y_pred2, target_names=class_names))

    plot_metrics(precision1, recall1, f1_1, precision2, recall2, f1_2, class_names)
    plot_confusion_matrix(y_true, y_pred1, y_pred2, class_names)
    plot_accuracy_graphs(model1_train_acc, model1_val_acc, model2_train_acc, model2_val_acc, num_epochs)

def build_data(lmdb_path):
    """Build Test Data from lmdb"""
    if REBUILD_DATA:
        image_path = os.path.join('data', 'test_set')
        test_data = ImageDataset(image_path=image_path, device=device, lmdb_path=lmdb_path, save_lmdb=True, mode="test")
    
        REBUILD_DATA = False
    else:
        test_data = ImageDataset(image_path=None, device=device, lmdb_path=lmdb_path, save_lmdb=False, mode="test")

    return test_data

def main():
    # Requirements
    IMG_SIZE = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    REBUILD_DATA = False

    # Loading test data
    lmdb_path = os.path.join('lmdb')
    test_data = build_data(lmdb_path)

    # Creating dataloader
    batch_size = 32
    test_dl = DataLoader(test_data, batch_size, shuffle=False)

    # Loading model
    fft_model = FFTGoogle(apply_fft=True, device=device)
    reg_model = FFTGoogle(apply_fft=False, device=device)

    # Loading model parameters
    fft_model.load_model_dict(os.path.join('models', 'google', 'fft_google_model.pth'))
    reg_model.load_model_dict(os.path.join('models', 'google', 'reg_google_model.pth'))

    # Setting up metrics evaluation
    class_names = ["bacterial", "normal", "viral"]
    
    tr_fft_acc = np.load(os.path.join('models', 'google', 'tr_fft_accuracy.npy'))
    val_fft_acc = np.load(os.path.join('models', 'google', 'val_fft_accuracy.npy'))
    
    tr_reg_acc = np.load(os.path.join('models', 'google', 'tr_reg_accuracy.npy'))
    val_reg_acc = np.load(os.path.join('models', 'google', 'val_reg_accuracy.npy'))

    # Evaluation
    compare_models(fft_model, reg_model, test_dl, class_names,
                   tr_fft_acc, val_fft_acc,
                   tr_reg_acc, val_reg_acc, 
                   num_epochs, device)

num_epochs = len(tr_fft_acc)

if __name__ == "__main__":
    main()