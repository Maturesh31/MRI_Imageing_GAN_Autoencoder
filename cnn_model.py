import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class SimpleCNN(nn.Module):
    """
    Enhanced 2D Convolutional Neural Network with Squeeze-and-Excitation (SE)
    Channel Attention and Spatial Dropout for ASD detection from 224x224 sMRI slices.
    Shared across all ablation experiments (E0, E1, E2, E3).
    """
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        def conv_block(inc, outc, drop2d=0.0):
            layers = [
                nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
            ]
            if drop2d > 0:
                layers.append(nn.Dropout2d(drop2d))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.block1 = conv_block(in_channels, 32)
        self.block2 = conv_block(32, 64)
        
        self.block3 = conv_block(64, 128, drop2d=0.15)
        self.se3 = SEBlock(128)
        
        self.block4 = conv_block(128, 256, drop2d=0.20)
        self.se4 = SEBlock(256)
        
        self.block5 = conv_block(256, 512, drop2d=0.25)
        self.se5 = SEBlock(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.se3(self.block3(x))
        x = self.se4(self.block4(x))
        x = self.se5(self.block5(x))
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)

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
    print("Enhanced SimpleCNN Summary:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {out.shape}")
    print(f"  Probs shape: {probs.shape}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
