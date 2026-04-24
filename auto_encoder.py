from base import BaseModel
import torch
import torch.nn as nn

class AE_Vanilla(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]
        out_channels = in_channels # python uses deepcopy for int variable

        modules = []
        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        # build decoder
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        rec = self.decoder(z)
        rec = self.final_layer(rec)
        return rec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)