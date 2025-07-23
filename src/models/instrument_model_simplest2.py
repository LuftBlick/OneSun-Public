import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModel(nn.Module):
    """
    Minimal instrument model with 5th-order polynomial absolute calibration:
      1) 5th-order polynomial absolute calibration (pixel-dependent sensitivity)
      2) Filter wheel attenuation factors
      3) Spectral resampling (input_dim -> output_dim)
    """
    def __init__(self, input_dim=2048, fwpos_dim=4, output_dim=2000):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 1) Learnable 5th-order polynomial coefficients: c0 + c1*x + ... + c5*x^5
        self.poly_order = 5
        # Initialize to identity calibration: c0=1, others=0
        coeffs = torch.zeros(self.poly_order + 1)
        coeffs[0] = 1.0
        self.poly_coeffs = nn.Parameter(coeffs)

        # Pre-compute pixel positions in [-1, 1]
        pos = torch.linspace(-1.0, 1.0, input_dim)
        self.register_buffer('pixel_positions', pos)

        # 2) Filter wheel attenuation factors (log-scale)
        init_vals = torch.ones(fwpos_dim)
        self.log_combo_att = nn.Parameter(init_vals.log())

        # Store configuration
        self.config = {
            'input_dim': input_dim,
            'fwpos_dim': fwpos_dim,
            'output_dim': output_dim,
            'poly_order': self.poly_order
        }

    def forward(self, x: torch.Tensor, fwpos_values: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        # 1) Polynomial calibration
        # Build matrix [poly_order+1, input_dim] of pixel_positions^i
        pows = [self.pixel_positions.pow(i) for i in range(self.poly_order + 1)]
        X = torch.stack(pows, dim=0)
        # [batch, input_dim]
        calib = (self.poly_coeffs.unsqueeze(1) * X).sum(dim=0)
        x = x * calib

        # 2) Filter wheel attenuation
        log_att_sum = torch.sum(fwpos_values * self.log_combo_att, dim=1, keepdim=True)
        attenuation = torch.exp(log_att_sum)
        x = x * attenuation

        # 3) Spectral resampling to output_dim
        x = F.interpolate(
            x.unsqueeze(1), size=self.output_dim,
            mode='linear', align_corners=False
        ).squeeze(1)

        return x
