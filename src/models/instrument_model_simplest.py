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
        init_vals = torch.ones(fwpos_dim)
        assert init_vals.numel() == fwpos_dim
        #self.combo_att = nn.Parameter(init_vals)
        self.log_combo_att = nn.Parameter(init_vals.log())

        
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
        
        # 2. Apply absolute calibration (pixel-dependent sensitivity)
        x = x * self.abs_calibration
        
        # 3. Apply filter wheel attenuation
        # Compute attenuation factor from filter wheel positions
        # multiplicative attenuation via exp(sum(log_att * one_hot))
        # log_att_sum: [B] = fwpos_values * self.log_combo_att → sum over dim=1
        log_att_sum = torch.sum(fwpos_values * self.log_combo_att, dim=1, keepdim=True)
        attenuation = torch.exp(log_att_sum)    # [B,1]; e.g. if one-hot idx2=1, log_att_sum = log(0.1)

        
        x = x * attenuation

        # 5. Spectral resampling (2048 -> 2000)
        # Simple linear interpolation
        x = F.interpolate(
            x.unsqueeze(1),  # [batch, 1, input_dim]
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [batch, output_dim]
        
        return x


