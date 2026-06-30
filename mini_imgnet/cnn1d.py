import torch
import torch.nn as nn

class CNN1DFeatureExtractor(nn.Module):
    """
    1-D CNN for features extraction from NetFlow data.
    Architecture:
    Conv1d (k3, p0, s1) x 2 -> MaxPool1d (k2, s2) -> Conv1d (k3, p0, s1) x 2 -> AdaptiveMaxPool1d
    """

    def __init__(self, input_dim=31, output_dim=64):
        """
        Args:
            input_dim: Number of features (default 31)
            output_dim: Feature embedding dimension (default 64)
        """
        super(CNN1DFeatureExtractor, self).__init__()
        
        self.body = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            
            # Layer 2
            nn.Conv1d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Layer 4
            nn.Conv1d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Global Pooling
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        out = self.body(x)
        out = out.view(out.size(0), -1) # Flatten (64 dimensions)
        return out


class CNN1DConvNet(nn.Module):
    """
    Wrapper for CNN1DFeatureExtractor.
    """
    out_dim = 64

    def __init__(self):
        super(CNN1DConvNet, self).__init__()
        self.cnn = CNN1DFeatureExtractor(input_dim=31, output_dim=self.out_dim)

    def forward(self, x):
        features = self.cnn(x)
        return features

    def stem_param(self):
        for param in self.cnn.parameters():
            yield param

    def stem_param_named(self):
        for name, param in self.cnn.named_parameters():
            yield name, param
