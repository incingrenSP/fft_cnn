import torch
import torch.nn as nn
import torch.fft as fft
import torchvision
from torchvision import models

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FFTConvNet(nn.Module):
    def __init__(self, conv_layer, fft_filter=None):
        super().__init__()
        self.conv_layer = conv_layer
        self.fft_filter = fft_filter
        self.kernel_fft = None
    
    def fft_filter_def(self, fft_x, height, width):
        cht, cwt = height // 2, width // 2
        mask_radius = 30

        fy, fx = torch.meshgrid(
            torch.arange(0, height, device=fft_x.device),
            torch.arange(0, width // 2 + 1, device=fft_x.device),
            indexing='ij'
        )
        mask_area = torch.sqrt((fx - cwt) ** 2 + (fy - cht) ** 2)

        if self.fft_filter == 'high':
            mask = (mask_area > mask_radius).float()
        else:
            mask = (mask_area <= mask_radius).float()

        filtered_fft = fft_x * mask
        return filtered_fft

    def forward(self, x):        
        batch_size, in_channels, height, width = x.size()
        out_channels = self.conv_layer.out_channels

        x = x.to(torch.float32)
        
        fft_x = fft.rfft2(x)
        fft_x = fft.fftshift(fft_x, dim=(-2, -1))

        # Comment if precomputing
        if self.kernel_fft is None or self.kernel_fft.shape[-2:] != (height, width):
            kernel_fft = fft.rfft2(self.conv_layer.weight, s=(height, width))

        if self.fft_filter is not None:
            fft_x = self.fft_filter_def(fft_x, height, width)

        fft_output = fft_x.unsqueeze(1) * kernel_fft.unsqueeze(0)
        fft_output = torch.sum(fft_output, dim=2)

        spatial_output = fft.irfft2(fft_output, s=(height,width))

        if self.conv_layer.bias is not None:
            spatial_output += self.conv_layer.bias.view(1, -1, 1, 1)

        # print(f"Input shape: {x.shape}")
        # print(f"Output shape: {spatial_output.shape}")

        return spatial_output

class FFTModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def _change_layer(self, conv_layer):
        fft_conv = FFTConvNet(conv_layer, 'low')
        return fft_conv
        
    def change_model(self, model, criteria, kernel):
        replace_count = 0

        for name, module in model.named_modules():
            if name.startswith(criteria) and isinstance(module, nn.Conv2d):
                if module.kernel_size[0] > kernel:
                    replace_count += 1
                    fft_conv = self._change_layer(module).to(self.device)
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, attr_name, fft_conv)

        print("Total Layers replaced: ", replace_count)
        return model

    def save_model_dict(self, path, name):
        full_path = os.path.join(path, name)
        torch.save(self.model.state_dict(), full_path)

    def load_model_dict(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def forward(self, image):
        x = self.model(image)
        return x

class FFTAlex(FFTModel):
    def __init__(self, apply_fft=False, device="cpu", IMG_SIZE=128):
        super().__init__(device)
        net = models.alexnet(weights=None)
        net = self.create_model(net, IMG_SIZE, apply_fft)
        if apply_fft:
            net = self.change_model(net, "features", 3)

        self.model = net.to(device)

    def create_model(self, net, IMG_SIZE, apply_fft):
        if apply_fft:
            net.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(256)
            )
            num_features = self._get_ftrs(net, IMG_SIZE)
            net.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(num_features, out_features=512),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=512, out_features=128),
                    nn.ReLU(),
                    nn.Linear(in_features=128, out_features=3)
            )
        else:
            net.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            num_features = self._get_ftrs(net, IMG_SIZE)
            net.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_features, out_features=4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=3)
            )

        return net

    def _get_ftrs(self, model, IMG_SIZE):
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        features = model.features(dummy_input)
        features = model.avgpool(features)
        num_features = features.view(features.size(0), -1).size(1)
        return num_features

class FFTGoogle(FFTModel):
    def __init__(self, apply_fft=False, device="cpu"):
        super().__init__(device)
        net = models.googlenet(weights=None, init_weights=True)
        net = self.create_model(net)
        if apply_fft:
            net = self.change_model(net, "inception", 1)

        self.model = net.to(device)

    def create_model(self, net):
        net.aux1 = None
        net.aux2 = None
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 3)
        )
        return net