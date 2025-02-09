# Comparative Study of Object Detection Models using Fourier Transformation Techniques

## Introduction
This project is an attempt to visualize the prediction of a CNN-based architecutre, AlexNet and Google LeNet (InceptionV1). It uses PyTorch as its main library, and features Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for visualization of output of each of these models.

## Project Members and Contributors
Preeti Adhikari (ACE078BCT050)
Rohan Basnet (ACE078BCT053)
Sabal Gautam (ACE078BCT054)
Samir Pokharel (ACE078BCT058)

## Project Supervisor
Er. Laxmi Prasad Bhatt
Acaedemic Project Coordinator
Advanced College of Engineering and Management

## Objective
- The primary objective of this project is to produce a light-weight system using Fast Fourier Transform based CNN model that can efficiently produce high precision, accuracy, and recall.
- Secondly, the project also aims to visualize the inflence of a model's prediction through Grad-CAM.

## Dataset
The project uses RSNA dataset, freely available at their [website][http://rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018]. The dataset includes X-Ray images of lungs that have been devided into three classes: _Bacterial_, _Normal_, _Viral_.

The datasets have been preprocessed, augmented and utilized for the purpose of this project.

## Libraries and Development Environment
The project uses the following libraries:
1. PyTorch
2. MatPlotLib
3. Sci-kit Learn
4. Seaborn
5. Shutil
6. Lightning Memory-Mapped Database (lmdb)
7. Default Python Libraries

The system has been developed on Jupyter Notebook with Python 3.10 kernel. It also uses GPU to train the models (if available). The models have been trained on NVIDIA GTX 1050 or higher.

