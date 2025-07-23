import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModel(nn.Module):
    """
    A simplified instrument model:
    1) Normalize by integration time
    2) Apply absolute calibration
    3) Convolve with a Gaussian slit function (learnable width)
    4) Regrid to desired output dimension
    5) Add a learned per-filter bias via embedding

    Expects filter-wheel positions as one-hot floats; converts to indices internally.
    """
    def __init__(
        self,
        input_dim: int,
        fwpos_dim: int,
        output_dim: int,
        slit_kernel_size: int = 11,
        init_slit_sigma: float = 2.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Absolute calibration parameter (per input pixel)
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

        # Slit convolution: positions buffer + learnable sigma
        self.slit_kernel_size = slit_kernel_size
        half = (slit_kernel_size - 1) / 2
        positions = torch.arange(slit_kernel_size) - half
        self.register_buffer('slit_positions', positions)
        self.slit_sigma = nn.Parameter(torch.tensor(init_slit_sigma))

        # Embedding for per-filter bias
        self.fw_bias = nn.Embedding(fwpos_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        fwpos_values: torch.Tensor,
        t_int: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] raw spectrum
            fwpos_values: [batch, fwpos_dim] one-hot or soft floats
            t_int: [batch, input_dim] integration times per pixel or broadcastable
        Returns:
            out: [batch, output_dim] corrected and regridded spectrum with filter bias
        """
        # 1) Normalize with integration time
        x = x / t_int

        # 2) Absolute calibration
        x = x * self.abs_calibration

        # 3) Dynamic Gaussian slit convolution
        sigma = self.slit_sigma.clamp(min=1e-6)
        pos = self.slit_positions.to(x.device).to(x.dtype)
        kernel = torch.exp(-0.5 * (pos / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.slit_kernel_size)

        x = x.unsqueeze(1)
        padding = (self.slit_kernel_size - 1) // 2
        x = F.conv1d(x, kernel, padding=padding)
        x = x.squeeze(1)

        # 4) Regrid to output dimension
        x = F.interpolate(
            x.unsqueeze(1),
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        # 5) Convert one-hot to indices, then add per-filter bias
        # handle floats: take argmax per sample
        fw_idx = fwpos_values.argmax(dim=1).long()
        bias = self.fw_bias(fw_idx)
        return x + bias
