import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



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
        exp_tau = torch.exp(-self.tau_rayleigh.to(amf.device).view(1, -1) * amf.view(-1, 1))

        # Multiply base_le by the tau_rayleigh exponential term.
        # Now, reconstructed_L1 has shape (batch_size, wavelength_dim)
        reconstructed_L1 = base_le * exp_tau


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



# New Encoder that outputs a locked latent space (ones + noise)
class LockedEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(LockedEncoder, self).__init__()
        self.latent_dim = latent_dim
        # Add a dummy trainable parameter to ensure optimizer works
        # This parameter won't affect the output but will make training continue
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x): # x is the L1_spectrum from the instrument model
        batch_size = x.size(0)
        # Create a latent space of all ones
        latent_space = torch.ones(
            (batch_size, self.latent_dim),
            dtype=x.dtype,
            device=x.device
        )
        # Add very small Gaussian noise to latent space
        latent_space = latent_space + torch.randn_like(latent_space) * 1e-3
        # The dummy parameter doesn't affect the output
        _ = self.dummy_param
        return latent_space


# New Atmospheric Model using the LockedEncoder
class AtmosphericModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, xsec_list, le_list, tau_rayleigh):
        """
        Atmospheric model with a locked latent space (always ones + noise).
        Signature is compatible with the standard AtmosphericModel.

        Args:
            input_dim (int): The dimension of the input L1 spectrum (e.g., 2000 pixels).
                                   This is also used as the output_dim for the decoder.
            latent_dim (int): The dimension of the latent space (number of "gases" or columns in xsec_df).
            output_dim (int): The dimension of the output L1 spectrum. Ignored, input_dim is used.
            xsec_list (pd.DataFrame): DataFrame of cross-section vectors for each gas.
            le_list (pd.DataFrame): DataFrame for the extraterrestrial solar spectrum.
            tau_rayleigh (torch.Tensor or array-like): Tensor for Rayleigh optical depth.
        """
        super(AtmosphericModel, self).__init__()
        self.encoder = LockedEncoder(latent_dim)
        # The decoder's output_dim is the same as the model_input_dim
        self.decoder = AtmosphericDecoder(input_dim, xsec_list, le_list, tau_rayleigh)

        # Store config for potential saving/loading, similar to InstrumentModel
        self.config = {
            'model_type': 'AtmosphericModelLockedLatent',
            'model_input_dim': input_dim,
            'latent_dim': latent_dim,
            # Note: xsec_df, le_df, tau_rayleigh themselves are not stored in config
            # as they are complex data objects. Their characteristics (like latent_dim from xsec_df)
            # and the model_input_dim define the architecture.
        }

    def forward(self, L1_spectrum, amf):
        # L1_spectrum is the input from the instrument model
        # The encoder will ignore the content of L1_spectrum and output locked ones
        latent_space = self.encoder(L1_spectrum)
        reconstructed_L1 = self.decoder(latent_space, amf)
        return reconstructed_L1
