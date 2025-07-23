import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel for convolution.
    """
    # Make sure kernel_size is odd
    half = (kernel_size - 1) / 2
    positions = torch.arange(kernel_size, device=device, dtype=dtype) - half
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # Shape to [out_channels, in_channels, kernel_size]
    return kernel.view(1, 1, kernel_size)


class InstrumentModel(nn.Module):
    """
    A simplified instrument model:
    1) Normalize by integration time
    2) Apply absolute calibration
    3) Convolve with a Gaussian slit function
    4) Regrid to desired output dimension
    5) Integrate filter-wheel position via a final linear layer
    """
    def __init__(
        self,
        input_dim: int,
        fwpos_dim: int,
        output_dim: int,
        slit_kernel_size: int = 11,
        slit_sigma: float = 2.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Absolute calibration parameter (per pixel)
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

        # Create and register a fixed Gaussian slit kernel
        kernel = create_gaussian_kernel(
            slit_kernel_size,
            slit_sigma,
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        self.register_buffer('slit_kernel', kernel)

        # Final linear layer to integrate filter-wheel one-hot encoding
        self.linear = nn.Linear(output_dim + fwpos_dim, output_dim)

    def forward(self, x: torch.Tensor, fwpos_values: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] raw spectrum
            fwpos_values: [batch, fwpos_dim] one-hot encoding of filter wheel positions
            t_int: [batch, input_dim] integration times per pixel or broadcastable
        Returns:
            out: [batch, output_dim] corrected and regridded spectrum
        """
        # 1) Normalize with integration time
        #x = x / t_int
        # 2) Absolute calibration
        x = x * self.abs_calibration

        # 3) Slit convolution
        # Prepare for 1D conv: [batch, 1, input_dim]
        x = x.unsqueeze(1)
        padding = (self.slit_kernel.size(2) - 1) // 2
        x = F.conv1d(x, self.slit_kernel, padding=padding)
        x = x.squeeze(1)

        # 4) Regrid to output dimension
        x = F.interpolate(
            x.unsqueeze(1),  # [batch, 1, current_dim]
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        # 5) Integrate filter-wheel position
        # Concatenate and project
        x = torch.cat([x, fwpos_values], dim=1)
        out = self.linear(x)
        return out
