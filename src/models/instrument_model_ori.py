import torch
import torch.nn as nn


class InstrumentModel(nn.Module):
    def __init__(self, input_dim, fwpos_dim, output_dim):
        super(InstrumentModel, self).__init__()
        # Define the network layers
        self.l1 = nn.Sequential(nn.Linear(input_dim + fwpos_dim, 2000))

        # Polynomial coefficients for non-linearity and wavelength dispersion
        self.poly_coeffs = nn.Parameter(torch.randn(input_dim, 2))

        # Adjusted convolution layer for both instrumental response and stray light correction
        kernel_size = 5  # Example, adjust based on your stray light effect
        self.convolution = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)

        # Fixed structure in dark current
        self.prnnu = nn.Parameter(torch.randn(input_dim))

        # Absolute calibration array
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

    def forward(self, x, fwpos_values, t_int):
        

        # PRNU correction
        x = x / self.prnnu

        # Normalize with integration times
        x = x / t_int
        # Absolute calibration
        x = x * self.abs_calibration
        # x now becomes (B, 1, input_dim)
        x = x.unsqueeze(1)

        # Apply the convolution (still (B, 1, input_dim))
        x = self.convolution(x)

        # Remove the channel dimension to go back to (B, input_dim)
        x = x.squeeze(1)

        # Concatenate the fwpos values along the feature dimension.
        # Make sure fwpos_values has shape (B, fwpos_dim)
        x = torch.cat((x, fwpos_values), dim=1)

        # Apply the fully-connected layer(s)
        x = self.l1(x)

        return x