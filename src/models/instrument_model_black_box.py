import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentModel(nn.Module):
    def __init__(
            self,
            input_dim=2000,  # Dimension of the spectrum
            fwpos_dim=0,  # Dimension of the fwpos tensor
            t_int_dim=1,  # Dimension of the t_int tensor (e.g., 1 if scalar)
            hidden_combined=1000,  # Dimension for the hidden layer
            output_dim=2000,  # Dimension of the output
            **kwargs  # Catches unused parameters from previous signature
    ):
        super(InstrumentModel, self).__init__()
        self.input_dim = input_dim
        self.fwpos_dim = fwpos_dim
        self.t_int_dim = t_int_dim
        self.hidden_dim = hidden_combined  # Use hidden_combined for the hidden layer size
        self.output_dim = output_dim

        if self.input_dim <= 0:
            raise ValueError("input_dim (spectrum dimension) must be greater than 0.")

        # Calculate the total input dimension for the first linear layer
        concatenated_input_dim = self.input_dim
        concatenated_input_dim += self.fwpos_dim


        self.fc1 = nn.Linear(concatenated_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, spectrum: torch.Tensor, fwpos: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        inputs_to_concatenate = [spectrum, fwpos]
        combined_input = torch.cat(inputs_to_concatenate, dim=1)
        x = self.fc1(combined_input)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x