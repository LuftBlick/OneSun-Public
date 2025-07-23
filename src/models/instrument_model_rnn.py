import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModelRNN(nn.Module):
    """
    Instrument model with an RNN (GRU) to model sequential dependencies in the spectrum.
    It includes:
    1. Absolute calibration (pixel-dependent sensitivity)
    2. Filter wheel attenuation factors
    3. A bidirectional GRU to model the instrument line shape
    4. Spectral resampling (2048 -> 2000 pixels)
    """

    def __init__(self, input_dim=2048, fwpos_dim=4, output_dim=2000, rnn_hidden_size=64, rnn_num_layers=2):
        super(InstrumentModelRNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        # 1. Absolute calibration - pixel-dependent sensitivity
        self.abs_calibration = nn.Parameter(torch.ones(input_dim))

        # 2. Filter wheel attenuation factors
        init_vals = torch.ones(fwpos_dim)
        self.log_combo_att = nn.Parameter(init_vals.log())

        # 3. RNN for modeling instrument line shape
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Linear layer to map RNN output back to a single feature per pixel
        self.fc = nn.Linear(rnn_hidden_size * 2, 1)  # *2 for bidirectional

        # Store config for saving/loading
        self.config = {
            'input_dim': input_dim,
            'fwpos_dim': fwpos_dim,
            'output_dim': output_dim,
            'rnn_hidden_size': rnn_hidden_size,
            'rnn_num_layers': rnn_num_layers
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
        # 1. Apply absolute calibration
        x = x * self.abs_calibration

        # 2. Apply filter wheel attenuation
        log_att_sum = torch.sum(fwpos_values * self.log_combo_att, dim=1, keepdim=True)
        attenuation = torch.exp(log_att_sum)
        x = x * attenuation

        # 3. Apply RNN to model spectral features
        # Reshape for RNN: [batch, seq_len, input_size]
        rnn_in = x.unsqueeze(-1)  # [batch, input_dim, 1]

        # Pass through GRU
        rnn_out, _ = self.rnn(rnn_in)  # [batch, input_dim, rnn_hidden_size * 2]

        # Map back to spectral correction
        spectral_correction = self.fc(rnn_out).squeeze(-1)  # [batch, input_dim]

        # Apply as a residual connection
        x = x + spectral_correction

        # 4. Spectral resampling (2048 -> 2000)
        x = F.interpolate(
            x.unsqueeze(1),  # [batch, 1, input_dim]
            size=self.output_dim,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [batch, output_dim]

        return x