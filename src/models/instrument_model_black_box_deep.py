import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModel(nn.Module):
    def __init__(
            self,
            input_dim=2000,  # Dimension of the spectrum
            fwpos_dim=0,  # Dimension of the fwpos tensor
            t_int_dim=1,  # Dimension of the t_int tensor (e.g., 1 if scalar)
            hidden_combined=1000,  # Dimension for the hidden layers
            output_dim=2000,  # Dimension of the output
            num_hidden_layers=4,  # Number of hidden layers (excluding input/output)
            **kwargs  # Catches unused parameters from previous signature
    ):
        super(InstrumentModel, self).__init__()
        self.input_dim = input_dim
        self.fwpos_dim = fwpos_dim
        self.t_int_dim = t_int_dim
        self.hidden_dim = hidden_combined
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        if self.input_dim <= 0:
            raise ValueError("input_dim (spectrum dimension) must be greater than 0.")

        # Calculate the total input dimension for the first linear layer
        concatenated_input_dim = self.input_dim + self.fwpos_dim

        # Input layer
        self.fc_in = nn.Linear(concatenated_input_dim, self.hidden_dim)
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)
        ])
        # Output layer
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, spectrum: torch.Tensor, fwpos: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        inputs_to_concatenate = [spectrum, fwpos]
        combined_input = torch.cat(inputs_to_concatenate, dim=1)
        x = self.fc_in(combined_input)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x

