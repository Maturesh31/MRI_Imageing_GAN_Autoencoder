import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 2D sMRI Brain Slice Enhancement / Denoising.
    Compresses 224x224 grayscale brain MRI slices into a compact latent bottleneck
    and reconstructs enhanced structural features, filtering out high-frequency noise.
    """
    def __init__(self, in_channels=1):
        super(ConvAutoencoder, self).__init__()

        # Encoder: (1, 224, 224) -> (512, 7, 7)
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 7x7 Bottleneck
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder: (512, 7, 7) -> (1, 224, 224)
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # Reconstructed image normalized to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def enhance(self, x):
        """Pass input image through autoencoder to produce enhanced/denoised version."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

if __name__ == "__main__":
    ae = ConvAutoencoder()
    dummy = torch.randn(4, 1, 224, 224)
    out = ae(dummy)
    print("Autoencoder Test:")
    print("  Input shape:", dummy.shape)
    print("  Output shape:", out.shape)
    params = sum(p.numel() for p in ae.parameters())
    print(f"  Trainable parameters: {params:,}")
