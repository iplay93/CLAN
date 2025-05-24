import torch
from torch import nn
from torch.nn import functional as F
from .Transformer import TransformerEncoder, TransformerEncoderLayer

class TFC(nn.Module):
    """
    TFC: Dual-view Transformer-based Encoder for Contrastive Representation Learning.
    
    - Time and Frequency domain Transformer encoders.
    - Each branch includes: Transformer → Flatten → Projector → Shift Classifier.
    - Returns both raw and attention-weighted representations.
    """
    def __init__(self, configs, args):
        super(TFC, self).__init__()

        # Time-domain Transformer Encoder
        self.encoder_t = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                dim_feedforward=2 * configs.input_channels,
                nhead=1,
                batch_first=True
            ),
            num_layers=2
        )

        # Frequency-domain Transformer Encoder
        self.encoder_f = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                dim_feedforward=2 * configs.input_channels,
                nhead=1,
                batch_first=True
            ),
            num_layers=2
        )

        # Time-domain projector and shift classifier
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift)

        # Frequency-domain projector and shift classifier
        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_f = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift_f)

        # Optional classifier for time features
        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, configs.num_classes)

    def forward(self, x_in_t, x_in_f):
        """
        Args:
            x_in_t: [B, C, L] time-domain input
            x_in_f: [B, C, L] frequency-domain input

        Returns:
            h_time: Flattened time-domain features
            z_time: Projected time-domain embedding
            s_time: Time-domain shift prediction
            h_freq: Flattened frequency-domain features
            z_freq: Projected frequency-domain embedding
            s_freq: Frequency-domain shift prediction
            x_weighted_t_all: List of attention-weighted time representations
            x_weighted_f_all: List of attention-weighted frequency representations
        """

        # Time domain
        x_in_t = x_in_t.permute(0, 2, 1)  # [B, L, D]
        x_t, attn_maps_t = self.encoder_t(x_in_t.float())
        h_time = x_t.reshape(x_t.size(0), -1)
        z_time = self.projector_t(h_time)
        s_time = self.shift_cls_layer_t(h_time)

        # Attention-weighted representations (time)
        x_weighted_t_all = [
            torch.bmm(attn.to(dtype=x_in_t.dtype, device=x_in_t.device), x_in_t)
            for attn in attn_maps_t
        ]

        # Frequency domain
        x_in_f = x_in_f.permute(0, 2, 1)  # [B, L, D]
        x_f, attn_maps_f = self.encoder_f(x_in_f.float())
        h_freq = x_f.reshape(x_f.size(0), -1)
        z_freq = self.projector_f(h_freq)
        s_freq = self.shift_cls_layer_f(h_freq)

        # Attention-weighted representations (frequency)
        x_weighted_f_all = [
            torch.bmm(attn.to(dtype=x_in_f.dtype, device=x_in_f.device), x_in_f)
            for attn in attn_maps_f
        ]

        return (
            h_time, z_time, s_time,
            h_freq, z_freq, s_freq,
            x_weighted_t_all, x_weighted_f_all
        )
