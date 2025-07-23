import torch
import torch.nn as nn

class InstrumentModel(nn.Module):
    def __init__(self, input_dim, fwpos_dim, output_dim):
        super().__init__()
        # Positional embedding (sin/cos) so the model knows pixel order
        pe = torch.linspace(0, 1, steps=input_dim, device='cpu')
        self.register_buffer(
            "pos_emb",
            torch.stack([pe.sin(), pe.cos()], dim=0)  # shape [2, input_dim]
        )

        # Higher-degree polynomial head (up to x^4)
        self.poly_coeffs = nn.Parameter(torch.randn(input_dim, 5))

        # PRNU (dark current) correction + absolute calibration
        self.prnnu           = nn.Parameter(torch.randn(input_dim))
        self.abs_calibration = nn.Parameter(torch.randn(input_dim))

        # Two-layer CNN to learn slow baselines
        kernel_size = 9
        self.conv1 = nn.Conv1d(3, 16, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act   = nn.ReLU()

        # Final dense head (as before)
        self.l1 = nn.Sequential(
            nn.Linear(input_dim + fwpos_dim, 2000),
            nn.ReLU(),
        )

    def forward(self, x, fwpos_values, t_int):
        """
        x:               [batch, input_dim] raw spectrum
        fwpos_values:    [batch, fwpos_dim] one-hot filter-wheel encoding
        """
        # 1) Polynomial non-linearity (x^1 … x^4)
        powers = torch.arange(1, 6, device=x.device, dtype=x.dtype)  # [1,2,3,4,5]
        x_poly = torch.sum(self.poly_coeffs * x.unsqueeze(-1).pow(powers), dim=-1)
        x = x + x_poly

        # 2) PRNU & absolute calibration
        x = x - self.prnnu
        # Normalize with integration times
        x = x / t_int
        
        x = x * self.abs_calibration

        # 3) Build conv input: [batch, 3, input_dim]
        pe = self.pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)
        x_in = x.unsqueeze(1)
        x_conv = torch.cat([x_in, pe], dim=1)

        # 4) Two-layer CNN baseline correction
        x_conv = self.act(self.conv1(x_conv))
        x_conv = self.conv2(x_conv).squeeze(1)

        # 5) Final FC head
        x_final = torch.cat([x_conv, fwpos_values], dim=1)
        out     = self.l1(x_final)

        return out
