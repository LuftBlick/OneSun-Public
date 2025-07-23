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
        # Store the data
        self.xsec_df = xsec_df          # cross-section DataFrame
        self.Le = le_df                 # extraterrestrial solar spectrum DataFrame
        # Fixed Rayleigh scattering optical depth (per-channel)
        self.tau_rayleigh = torch.tensor(
            tau_rayleigh, dtype=torch.float32
        )

        # Dummy spectral coordinate 't' for polynomial basis
        n_channels = self.tau_rayleigh.numel()
        self.register_buffer('t', torch.linspace(0.0, 1.0, n_channels))

        # Learnable 3rd-order polynomial coefficients for aerosol/cloud layer
        # tau_base = c0 + c1*t + c2*t^2 + c3*t^3
        self.aero_coeffs = nn.Parameter(torch.zeros(4))

        # Cloud presence factor: maps latent features -> [0,1]
        # LazyLinear infers in_features at first forward pass, ensuring backward compatibility
        self.cloud_net = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, latent_space, amf):
        batch_size = amf.size(0)
        # Base extraterrestrial spectrum
        base_le = torch.tensor(
            self.Le.iloc[:, 0].values,
            dtype=latent_space.dtype,
            device=latent_space.device
        ).view(1, -1).repeat(batch_size, 1)

        # Rayleigh extinction
        exp_ray = torch.exp(
            -self.tau_rayleigh.view(1, -1) * amf.view(-1, 1)
        )
        reconstructed_L1 = base_le * exp_ray

        # Compute cloud strength per sample (0=no cloud, 1=full)
        cloud_strength = self.cloud_net(latent_space)  # (batch,1)

        # Polynomial spectral shape (constant across batch)
        t = self.t.view(1, -1).to(latent_space.device)
        c0, c1, c2, c3 = self.aero_coeffs
        tau_base = c0 + c1 * t + c2 * (t ** 2) + c3 * (t ** 3)  # (1, channels)

        # Per-sample cloud optical depth and extinction
        tau_aero = cloud_strength.view(-1, 1) * tau_base  # (batch, channels)
        exp_aero = torch.exp(
            -tau_aero * amf.view(-1, 1)
        )
        reconstructed_L1 = reconstructed_L1 * exp_aero

        # Molecular absorption by gases
        for i, column_name in enumerate(self.xsec_df.columns):
            xsec = torch.tensor(
                self.xsec_df[column_name].values,
                dtype=latent_space.dtype,
                device=latent_space.device
            )
            latent_val = latent_space[:, i:i+1]
            gas_factor = torch.exp(-latent_val * xsec * amf.view(-1, 1))
            reconstructed_L1 = reconstructed_L1 * gas_factor

        # Ensure non-negativity
        return F.relu(reconstructed_L1)


class AtmosphericModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, xsec_list, le_list, tau_rayleigh):
        super(AtmosphericModel, self).__init__()
        self.encoder = AtmosphericEncoder(input_dim, latent_dim)
        self.decoder = AtmosphericDecoder(output_dim, xsec_list, le_list, tau_rayleigh)

    def forward(self, L1_spectrum, amf):
        latent_space = self.encoder(L1_spectrum)
        reconstructed_L1 = self.decoder(latent_space, amf)
        return reconstructed_L1