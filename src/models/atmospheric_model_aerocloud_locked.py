import torch
import torch.nn as nn
import torch.nn.functional as F

# Locked encoder: always outputs ones + small noise
class LockedEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(LockedEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        latent_space = torch.ones(
            (batch_size, self.latent_dim),
            dtype=x.dtype,
            device=x.device
        )
        latent_space = latent_space + torch.randn_like(latent_space) * 1e-3
        _ = self.dummy_param
        return latent_space

# Decoder from aerocloud (with cloud/aerosol logic)
class AtmosphericDecoder(nn.Module):
    def __init__(self, output_dim, xsec_df, le_df, tau_rayleigh):
        super(AtmosphericDecoder, self).__init__()
        self.xsec_df = xsec_df
        self.Le = le_df
        self.tau_rayleigh = torch.tensor(
            tau_rayleigh, dtype=torch.float32
        )
        n_channels = self.tau_rayleigh.numel()
        self.register_buffer('t', torch.linspace(0.0, 1.0, n_channels))
        self.aero_coeffs = nn.Parameter(torch.zeros(4))
        self.cloud_net = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, latent_space, amf):
        batch_size = amf.size(0)
        base_le = torch.tensor(
            self.Le.iloc[:, 0].values,
            dtype=latent_space.dtype,
            device=latent_space.device
        ).view(1, -1).repeat(batch_size, 1)
        exp_ray = torch.exp(
            -self.tau_rayleigh.view(1, -1) * amf.view(-1, 1)
        )
        reconstructed_L1 = base_le * exp_ray
        cloud_strength = self.cloud_net(latent_space)
        t = self.t.view(1, -1).to(latent_space.device)
        c0, c1, c2, c3 = self.aero_coeffs
        tau_base = c0 + c1 * t + c2 * (t ** 2) + c3 * (t ** 3)
        tau_aero = cloud_strength.view(-1, 1) * tau_base
        exp_aero = torch.exp(
            -tau_aero * amf.view(-1, 1)
        )
        reconstructed_L1 = reconstructed_L1 * exp_aero
        for i, column_name in enumerate(self.xsec_df.columns):
            xsec = torch.tensor(
                self.xsec_df[column_name].values,
                dtype=latent_space.dtype,
                device=latent_space.device
            )
            latent_val = latent_space[:, i:i+1]
            gas_factor = torch.exp(-latent_val * xsec * amf.view(-1, 1))
            reconstructed_L1 = reconstructed_L1 * gas_factor
        return F.relu(reconstructed_L1)

class AtmosphericModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, xsec_list, le_list, tau_rayleigh):
        super(AtmosphericModel, self).__init__()
        self.encoder = LockedEncoder(latent_dim)
        self.decoder = AtmosphericDecoder(output_dim, xsec_list, le_list, tau_rayleigh)

    def forward(self, L1_spectrum, amf):
        latent_space = self.encoder(L1_spectrum)
        reconstructed_L1 = self.decoder(latent_space, amf)
        return reconstructed_L1

