import torch
from torch import nn
from torch.nn import functional as F
from .Transformer import TransformerEncoder, TransformerEncoderLayer


class TFT_VIZ(nn.Module):
    """
    TFT_VIZ: Time-Frequency Transformer Encoder with Attention Visualization Support.

    This model extracts representations from both time-domain and frequency-domain inputs 
    using separate Transformer encoders. In addition to raw features, it outputs:
    - Projected embeddings for contrastive learning
    - Auxiliary shift classification logits
    - Attention-weighted feature representations (useful for visualization)
    """

    def __init__(self, configs, args):
        super(TFT_VIZ, self).__init__()

        # Time-domain Transformer encoder
        self.encoder_t = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                nhead=1,
                dim_feedforward=2 * configs.input_channels,
                batch_first=True
            ),
            num_layers=2
        )

        # Frequency-domain Transformer encoder
        self.encoder_f = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                nhead=1,
                dim_feedforward=2 * configs.input_channels,
                batch_first=True
            ),
            num_layers=2
        )

        # Projector for time-domain encoder output (flattened)
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Auxiliary shift classification head for time-domain features
        self.shift_cls_layer_t = nn.Linear(
            configs.TSlength_aligned * configs.input_channels,
            args.K_shift
        )

        # Projector for frequency-domain encoder output
        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Auxiliary shift classification head for frequency-domain features
        self.shift_cls_layer_f = nn.Linear(
            configs.TSlength_aligned * configs.input_channels,
            args.K_shift_f
        )

        # Optional classifier for task-specific training (e.g., activity classification)
        self.linear = nn.Linear(
            configs.TSlength_aligned * configs.input_channels,
            configs.num_classes
        )

    def forward(self, x_in_t, x_in_f):
        """
        Forward pass through both time and frequency branches.

        Args:
            x_in_t (Tensor): Time-domain input tensor of shape [B, C, L]
            x_in_f (Tensor): Frequency-domain input tensor of shape [B, C, L]

        Returns:
            h_time (Tensor): Flattened time-domain features
            z_time (Tensor): Projected time-domain embedding (for contrastive learning)
            s_time (Tensor): Shift classification logits (time-domain)
            h_freq (Tensor): Flattened frequency-domain features
            z_freq (Tensor): Projected frequency-domain embedding
            s_freq (Tensor): Shift classification logits (frequency-domain)
            x_weighted_t_all (List[Tensor]): List of attention-weighted time features per layer
            x_weighted_f_all (List[Tensor]): List of attention-weighted frequency features per layer
        """

        # --- Time-domain branch ---
        x_in_t = x_in_t.permute(0, 2, 1)  # [B, L, C]
        x_t, attn_maps_t = self.encoder_t(x_in_t.float())  # [B, L, C], [N_layers × B × L × L]
        h_time = x_t.reshape(x_t.size(0), -1)  # Flatten
        z_time = self.projector_t(h_time)
        s_time = self.shift_cls_layer_t(h_time)

        # Compute attention-weighted features for time
        x_weighted_t_all = [
            torch.bmm(attn.to(dtype=x_in_t.dtype, device=x_in_t.device), x_in_t)
            for attn in attn_maps_t
        ]

        # --- Frequency-domain branch ---
        x_in_f = x_in_f.permute(0, 2, 1)  # [B, L, C]
        x_f, attn_maps_f = self.encoder_f(x_in_f.float())
        h_freq = x_f.reshape(x_f.size(0), -1)
        z_freq = self.projector_f(h_freq)
        s_freq = self.shift_cls_layer_f(h_freq)

        # Compute attention-weighted features for frequency
        x_weighted_f_all = [
            torch.bmm(attn.to(dtype=x_in_f.dtype, device=x_in_f.device), x_in_f)
            for attn in attn_maps_f
        ]

        return (
            h_time, z_time, s_time,
            h_freq, z_freq, s_freq,
            x_weighted_t_all, x_weighted_f_all
        )
