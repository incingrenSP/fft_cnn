import torch
import torch.nn as nn
import torch.fft as fft

class FFTConvNet(nn.Module):
    def __init__(self, conv_layer, fft_filter=None):
        super().__init__()
        self.conv_layer = conv_layer  # Original Conv2d layer
        self.fft_filter = fft_filter

    def fft_filter_def(self, fft_x, height, width):
        cht, cwt = height // 2, width // 2
        mask_radius = 30

        # Create a meshgrid for the mask
        fy, fx = torch.meshgrid(
            torch.arange(0, height, device=fft_x.device),
            torch.arange(0, width, device=fft_x.device),
            indexing='ij'
        )
        mask_area = torch.sqrt((fx - cwt) ** 2 + (fy - cht) ** 2)

        # Create the mask based on the filter type
        if self.fft_filter == 'high':
            mask = (mask_area > mask_radius).float()
        else:
            mask = (mask_area <= mask_radius).float()

        # Apply the mask to the FFT of the input
        filtered_fft = fft_x * mask
        return filtered_fft

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        out_channels = self.conv_layer.out_channels  # Number of output channels

        # Apply FFT on input image
        fft_x = fft.fft2(x)  # Shape: [batch_size, in_channels, height, width]
        fft_x = fft.fftshift(fft_x)

        # Apply FFT on the convolutional kernel
        kernel_fft = fft.fft2(self.conv_layer.weight, s=(height, width))  # Shape: [out_channels, in_channels, height, width]
        kernel_fft = fft.fftshift(kernel_fft)

        # Apply FFT filter (low-pass or high-pass)
        if self.fft_filter is not None:
            fft_x = self.fft_filter_def(fft_x, height, width)

        # Perform element-wise complex multiplication
        fft_output = fft_x.unsqueeze(1) * kernel_fft.unsqueeze(0)  # Broadcast multiplication
        fft_output = torch.sum(fft_output, dim=2)  # Sum over input channels

        # Apply inverse FFT
        fft_output = fft.ifftshift(fft_output, dim=(-2, -1))
        spatial_output = fft.ifft2(fft_output, dim=(-2, -1)).real  # Shape: [batch_size, out_channels, height, width]

        # Add bias (if applicable)
        if self.conv_layer.bias is not None:
            spatial_output += self.conv_layer.bias.view(1, -1, 1, 1)

        # Debug: Print shapes
        # print(f"Input shape: {x.shape}")
        # print(f"Output shape: {spatial_output.shape}")

        # Return output
        return spatial_output

# Function to replace Conv2d with FFTConvNet
def change_layer(layer):
    fft_conv = FFTConvNet(layer, 'low')
    return fft_conv