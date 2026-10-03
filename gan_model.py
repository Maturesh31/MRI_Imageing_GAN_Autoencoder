import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    """
    Conditional Generator for 2D sMRI brain slices (1, 224, 224).
    Maps latent vector z + class label (0=Control, 1=ASD) -> 224x224 grayscale brain slice.
    """
    def __init__(self, latent_dim=128, num_classes=2, embed_dim=32):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        in_dim = latent_dim + embed_dim

        # Initial projection: 256 channels of 7x7
        self.init_dense = nn.Sequential(
            nn.Linear(in_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling blocks: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # Output range in [0, 1] matching min-max normalized sMRI
        )

    def forward(self, z, labels):
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        x = self.init_dense(x)
        x = x.view(-1, 256, 7, 7)
        img = self.conv_blocks(x)
        return img


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator for 2D sMRI brain slices (1, 224, 224).
    Determines whether a brain slice is real or synthetic given its class label.
    """
    def __init__(self, num_classes=2, img_size=224):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

        # Input: 1 (image) + 1 (label map) = 2 channels
        self.features = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1)  # Logits
        )

    def forward(self, img, labels):
        # Create spatial label map matching image size
        c_map = self.label_embed(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, c_map], dim=1)
        feat = self.features(x)
        validity = self.classifier(feat)
        return validity.squeeze(-1)

if __name__ == "__main__":
    G = ConditionalGenerator()
    D = ConditionalDiscriminator()
    z = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    fake_imgs = G(z, labels)
    validity = D(fake_imgs, labels)
    print("Conditional GAN Test:")
    print("  Generated images shape:", fake_imgs.shape)
    print("  Discriminator output shape:", validity.shape)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"  Generator params: {g_params:,} | Discriminator params: {d_params:,}")
