"""CNN + Spatial LSTM Neural Architecture for Road Video Damage Segmentation."""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CNNSpatialLSTM(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        # Spatial Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Temporal ConvLSTM
        self.conv_lstm = ConvLSTMCell(in_channels=64, hidden_channels=hidden_dim)
        # Decoder to binary mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        outputs = []
        h_state = None
        c_state = None

        for step in range(t):
            frame = x[:, step]
            feats = self.encoder(frame)
            if h_state is None:
                h_state = torch.zeros_like(feats)
                c_state = torch.zeros_like(feats)
            h_state, c_state = self.conv_lstm(feats, h_state, c_state)
            mask_logit = self.decoder(h_state)
            outputs.append(mask_logit)

        return torch.stack(outputs, dim=1)  # (B, T, 1, H, W)
