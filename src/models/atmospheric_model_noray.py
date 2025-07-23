import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the encoder part of the atmospheric model
class AtmosphericEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AtmosphericEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        latent_space = F.leaky_relu(self.fc2(x))
        return latent_space


class AtmosphericDecoder(nn.Module):
    def __init__(self, output_dim, xsec_df, le_df, tau_rayleigh):
        super(AtmosphericDecoder, self).__init__()
        # Store the cross-sections (xsec) and extraterrestrial solar spectrum (Le)
        self.xsec_df = xsec_df  # DataFrame of cross-section vectors for each gas
        self.Le = le_df  # DataFrame (or similar) for the extraterrestrial solar spectrum
        # Create a constant tensor for tau_rayleigh
        self.tau_rayleigh = torch.tensor(tau_rayleigh, dtype=torch.float32)

    def forward(self, latent_space, amf):
        batch_size = amf.size(0)

        # Create the base extraterrestrial spectrum tensor.
        # Originally, this was shape (1, wavelength_dim)
        # We use .repeat to get a shape of (batch_size, wavelength_dim)
        base_le = torch.tensor(
            self.Le.iloc[:, 0].values,
            dtype=latent_space.dtype,
            device=latent_space.device
        ).view(1, -1).repeat(batch_size, 1)

        # Compute the exponential term for tau_rayleigh.
        # amf.view(-1, 1) makes sure amf is of shape (batch_size, 1)
        #exp_tau = torch.exp(-self.tau_rayleigh.view(1, -1) * amf.view(-1, 1))

        # Multiply base_le by the tau_rayleigh exponential term.
        # Now, reconstructed_L1 has shape (batch_size, wavelength_dim)
        reconstructed_L1 = base_le #* exp_tau


        # Loop through each gas/column in the cross-section DataFrame
        for i, column_name in enumerate( self.xsec_df.columns):
            # Create a tensor for the cross-section of this gas.
            xsec = torch.tensor(
                self.xsec_df[column_name].values,
                dtype=latent_space.dtype,
                device=latent_space.device
            )
            # Extract the corresponding latent parameter.
            # latent_space[:, i:i+1] has shape (batch_size, 1)
            latent_value = latent_space[:, i:i + 1]

            # Compute the exponential term for the current gas.
            # This term is of shape (batch_size, wavelength_dim)
            gas_factor = torch.exp(-latent_value * xsec * amf.view(-1, 1))

            # Multiply the current gas factor.
            # We avoid in-place modification issues by reassigning.
            reconstructed_L1 = reconstructed_L1 * gas_factor

        # Optionally, apply a ReLU to ensure non-negativity.
        reconstructed_L1 = F.relu(reconstructed_L1)

        return reconstructed_L1


class AtmosphericModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, xsec_list, le_list, tau_rayleigh):
        super(AtmosphericModel, self).__init__()
        self.encoder = AtmosphericEncoder(input_dim, latent_dim)
        self.decoder = AtmosphericDecoder(output_dim, xsec_list, le_list, tau_rayleigh)

    def forward(self, L1_spectrum, amf):
        latent_space = self.encoder(L1_spectrum)
        reconstructed_L1 = self.decoder(latent_space, amf)
        return reconstructed_L1