import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Standard 2D Convolutional Neural Network for ASD detection from 224x224 sMRI slices.
    Shared across all ablation experiments (E0, E1, E2, E3).
    """
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        # Conv Block 1: (1, 224, 224) -> (32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2: (32, 112, 112) -> (64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3: (64, 56, 56) -> (128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 4: (128, 28, 28) -> (256, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Pooling & Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)  # Output shape: (B,)

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(4, 1, 224, 224)
    out = model(x)
    probs = model.predict_proba(x)
    print("SimpleCNN Summary:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {out.shape}")
    print(f"  Probs shape: {probs.shape}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
