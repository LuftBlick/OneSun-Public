import torch
import torch.nn as nn
import torch.nn.functional as F


class DispersionLayer(nn.Module):
    """Applies dispersion shifts to each wavelength before convolution"""
    def __init__(self, input_dim):
        super().__init__()
        self.dispersion = nn.Parameter(torch.zeros(input_dim))  # Trainable shift per wavelength

    def forward(self, x):
        """
        x: (batch_size, num_wavelengths)
        Returns: (batch_size, num_wavelengths) after shifting wavelengths
        """
        batch_size, num_wavelengths = x.shape
        shifted_x = torch.zeros_like(x)

        for i in range(num_wavelengths):
            shift = self.dispersion[i]  # Learnable shift for this wavelength
            shifted_x[:, i] = self._interpolate_shift(x, shift, i)

        return shifted_x

    def _interpolate_shift(self, x, shift, i):
        """Interpolates a single wavelength point `i` to apply a fractional pixel shift."""
        left_index = max(i - 1, 0)
        right_index = min(i + 1, x.shape[1] - 1)
        return (1 - shift) * x[:, left_index] + shift * x[:, right_index]


class VariableSlitFunction(nn.Module):
    """Applies a learned wavelength-dependent transformation slit function"""
    def __init__(self, input_dim, kernel_size=11, n_points=50):
        super().__init__()

        self.kernel_size = kernel_size
        self.n_points = n_points

        # Instead of a fixed Gaussian, we now LEARN the transformation kernel per wavelength
        self.slit_kernels = nn.Parameter(torch.randn(input_dim, self.n_points))  # Trainable kernel

    def forward(self, x):
        num_w = x.shape[1]
        # normalize slit kernels
        slit_kernels = F.softmax(self.slit_kernels, dim=1).unsqueeze(1)
        # conv → (batch, num_w, out_width)
        conv_out = F.conv1d(
            x.unsqueeze(-1),
            slit_kernels,
            padding=self.n_points // 2,
            groups=num_w
        )
        # collapse the extra dim to get back to (batch, num_w)
        x = conv_out.mean(dim=-1)  # or use .sum(dim=-1), or pick center: conv_out[..., out_width//2]
        return x


class InstrumentModel(nn.Module):
    """Instrument Model with Dispersion Correction and Effective Slit Function"""
    def __init__(self, input_dim, fwpos_dim, output_dim, kernel_size=11, n_points=50):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Step 1: Apply Dispersion (Wavelength Shifts)
        self.dispersion_layer = DispersionLayer(input_dim)

        # Step 2: Apply Effective Slit Function
        self.slit_layer = VariableSlitFunction(input_dim, kernel_size, n_points)

        # Polynomial coefficients for non-linearity and wavelength dispersion
        self.poly_coeffs = nn.Parameter(torch.randn(input_dim, 2))

        # Convolution for stray light correction
        # AK : not used currectly, need to define better kernel  
        self.kernel_size = kernel_size
        self.convolution = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        # Fixed structure in dark current
        self.prnnu = nn.Parameter(torch.randn(input_dim))

        # Absolute calibration array
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

        # Final correction network (fwpos-dependent)
        self.l1 = nn.Sequential(
            nn.Linear(input_dim + fwpos_dim, 2000),
            nn.Sigmoid(),
        )

    def forward(self, x, fwpos_values, t_int):
        """Applies all instrument-specific corrections and transforms the spectrum to the ideal form"""

        # Apply Non-linearity Correction
        x = torch.sum(self.poly_coeffs * x.unsqueeze(-1) ** torch.arange(1, 3), dim=-1)

        # Apply PRNU Correction
        x = x / self.prnnu

        # Normalize with Integration Time
        x = x / t_int

        # Apply Absolute Calibration
        x = x * self.abs_calibration

        # Apply Dispersion Correction (Shifts Spectrum)
        x = self.dispersion_layer(x)

        # Apply Effective Slit Function (Corrects Real to Ideal)
        x = self.slit_layer(x)

        # AK: not needed anymore... Apply Instrumental Convolution (Final Correction)
        #x = self.convolution(x.unsqueeze(1)).squeeze(1)

        # Append fwpos values; ensure both tensors are 2-D
        fw = fwpos_values.view(x.size(0), -1)
        x = torch.cat([x, fw], dim=1)

        # Apply Final Fully Connected Correction
        # AK : not sure yet about this step !?!
        x = self.l1(x)

        return x
