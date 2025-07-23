import torch
import torch.nn as nn
import torch.nn.functional as F

def create_gaussian_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel for convolution.
    """
    half = (kernel_size - 1) / 2
    positions = torch.arange(kernel_size, device=device, dtype=dtype) - half
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # Shape to [out_channels, in_channels, kernel_size]
    return kernel.view(1, 1, kernel_size)


class InstrumentModel(nn.Module):
    """
    A minimal instrument model that:
      1) Applies absorbance calibration per pixel.
      2) Convolves with a fixed Gaussian slit function.
      3) Regrids from input_dim (e.g. 2048) to output_dim (e.g. 2000).
      4) Applies a fixed attenuation factor if filter‐wheel filters are engaged.
    
    Args:
        input_dim (int): Number of raw detector pixels (e.g. 2048).
        fwpos_dim (int): Number of one‐hot filter‐wheel entries (e.g. 2).
        output_dim (int): Desired output channels (e.g. 2000).
        slit_kernel_size (int): Size of the Gaussian kernel for convolution.
        slit_sigma (float): Standard deviation for the Gaussian slit.
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
        self.fwpos_dim = fwpos_dim

        # 1) Absolute calibration parameter (per raw pixel)
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

        # 2) Slit convolution: register a fixed Gaussian kernel
        kernel = create_gaussian_kernel(
            slit_kernel_size,
            slit_sigma,
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        self.register_buffer('slit_kernel', kernel)

        # 3) (Optional) A small learnable “sensitivity” per output pixel after regridding
        self.sensitivity = nn.Parameter(torch.ones(output_dim))

        # 4) Attenuation factors: one learnable scalar per filter‐wheel position
        #    If fwpos_values is a one‐hot of length fwpos_dim, then:
        #      attenuation = 1.0 - (fwpos_values @ attn_factors)
        #    We clamp the result to [0,1] in forward().
        self.attn_factors = nn.Parameter(torch.zeros(fwpos_dim))

    def forward(
        self,
        x: torch.Tensor,           # [batch, input_dim]
        fwpos_values: torch.Tensor,# [batch, fwpos_dim], one-hot indicating inserted filters
        t_int: torch.Tensor = None # [batch, input_dim] or broadcastable (ignored here)
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: raw spectrum, shape [B, input_dim].
            fwpos_values: one-hot filter-wheel encoding, shape [B, fwpos_dim].
                          Each index with “1” means that filter causes attenuation.
            t_int: integration time per pixel (not used in this minimal version).
        Returns:
            out: corrected spectrum, shape [B, output_dim].
        """
        B = x.shape[0]

        # 1) Absolute calibration: per-pixel multiplication
        #    x_cal has shape [B, input_dim]
        x_cal = x * self.abs_calibration  # :contentReference[oaicite:1]{index=1}

        # 2) Slit convolution
        #    - Expand to [B, 1, input_dim] so we can conv1d with a single Gaussian kernel.
        x_conv = x_cal.unsqueeze(1)  # → [B, 1, input_dim]
        padding = (self.slit_kernel.size(2) - 1) // 2
        x_conv = F.conv1d(x_conv, self.slit_kernel, padding=padding)  # → [B, 1, input_dim]
        x_conv = x_conv.squeeze(1)  # → [B, input_dim]

        # 3) Regrid (interpolate) from input_dim → output_dim (linear interpolation)
        x_regrid = F.interpolate(
            x_conv.unsqueeze(1),  # [B, 1, input_dim]
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # → [B, output_dim]

        # 3.5) Apply small per-output-pixel sensitivity if desired
        x_sens = x_regrid * self.sensitivity  # → [B, output_dim]

        # 4) Attenuation from filter‐wheel:
        #    Compute attenuation factor ∈ ℝ^{B} as:
        #       atten = 1.0 - (fwpos_values @ attn_factors), 
        #    then clamp to [0, 1].
        #
        #    If fwpos_values[i] = 1 for a given wheel, that wheel’s attn_factors[j] reduces total signal.
        atten = 1.0 - torch.matmul(fwpos_values, self.attn_factors)  # [B]
        atten = atten.clamp(min=0.0, max=1.0).view(B, 1)            # [B, 1]

        out = x_sens * atten  # broadcast → [B, output_dim]
        return out
