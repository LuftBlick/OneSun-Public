import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModel(nn.Module):
    """
    Minimal instrument model with only essential physics:
    1. Absolute calibration (pixel-dependent sensitivity)
    2. Filter wheel attenuation factors
    3. Simple spectral convolution (instrument line shape)
    4. Spectral resampling (2048 -> 2000 pixels)
    """
    
    def __init__(self, input_dim=2048, fwpos_dim=4, output_dim=2000):
        super(InstrumentModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1. Absolute calibration - pixel-dependent sensitivity
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))
        
        # 2. Filter wheel attenuation factors
        # initialize the 5 combo attenuations:
        init_vals = torch.tensor([1.0, 1, 0.5, 1.0, 1.0])
        assert init_vals.numel() == fwpos_dim
        #self.combo_att = nn.Parameter(init_vals)
        self.log_combo_att = nn.Parameter(init_vals.log())

        # 3. Simple convolution for instrument line shape (ILS)
        # Small kernel for spectral smearing
        kernel_size = 5  # Small kernel for minimal smearing
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        
        # Initialize convolution with a Gaussian-like kernel
        with torch.no_grad():
            # Create a simple Gaussian kernel
            kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1]).float()
            kernel = kernel / kernel.sum()  # Normalize
            self.conv.weight.data = kernel.view(1, 1, -1)
        
        # Store config for saving/loading
        self.config = {
            'input_dim': input_dim,
            'fwpos_dim': fwpos_dim,
            'output_dim': output_dim
        }
    
    def forward(self, x, fwpos_values, t_int):
        """
        Forward pass through the instrument model
        
        Args:
            x: Raw spectrum [batch, input_dim] (2048 pixels)
            fwpos_values: Filter wheel positions [batch, fwpos_dim] (one-hot encoded)
            t_int: Integration times [batch] (for normalization)
        
        Returns:
            calibrated spectrum [batch, output_dim] (2000 pixels)
        """
        # 1. Normalize by integration time
        #x = x / t_int.unsqueeze(-1)
        
        # 2. Apply absolute calibration (pixel-dependent sensitivity)
        x = x * self.abs_calibration
        
        # 3. Apply filter wheel attenuation
        # Compute attenuation factor from filter wheel positions
        # multiplicative attenuation via exp(sum(log_att * one_hot))
        # log_att_sum: [B] = fwpos_values * self.log_combo_att → sum over dim=1
        log_att_sum = torch.sum(fwpos_values * self.log_combo_att, dim=1, keepdim=True)
        attenuation = torch.exp(log_att_sum)    # [B,1]; e.g. if one-hot idx2=1, log_att_sum = log(0.1)

        
        x = x * attenuation
        
        # 4. Apply instrument line shape convolution
        # Reshape for 1D convolution: [batch, 1, input_dim]
        #x = x.unsqueeze(1)
        #x = self.conv(x)
        #x = x.squeeze(1)  # Back to [batch, input_dim]
        
        # 5. Spectral resampling (2048 -> 2000)
        # Simple linear interpolation
        x = F.interpolate(
            x.unsqueeze(1),  # [batch, 1, input_dim]
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [batch, output_dim]
        
        return x


