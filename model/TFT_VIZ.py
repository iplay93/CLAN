import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TFT_VIZ(nn.Module):
    """
    TFT_VIZ: Time-Frequency Transformer with Attention Visualization.

    Supports:
    - Time-only, Frequency-only, or Dual-view input
    - Projected features for contrastive learning
    - Shift classification (binary or multi-class)
    - Attention-weighted intermediate features for visualization
    """
    def __init__(self, configs, args):
        super(TFT_VIZ, self).__init__()

        self.mode = args.training_mode  # 'T', 'F', 'TF'
        self.use_binary_shift = getattr(args, "binary_shift", False)
        self.K_shift = getattr(args, "K_shift", 2)
        self.K_shift_f = getattr(args, "K_shift_f", 2)

        # Time-domain encoder
        self.encoder_t = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                nhead=1,
                dim_feedforward=2 * configs.input_channels,
                batch_first=True
            ),
            num_layers=2,
            return_attention=True
        )

        # Frequency-domain encoder
        self.encoder_f = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.input_channels,
                nhead=1,
                dim_feedforward=2 * configs.input_channels,
                batch_first=True
            ),
            num_layers=2,
            return_attention=True
        )

        # Flattened input dimension: [B, C, L] → flatten to C×L
        in_dim = configs.TSlength_aligned * configs.input_channels

        self.projector_t = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.projector_f = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.shift_cls_layer_t = nn.Linear(in_dim, 1 if self.use_binary_shift else self.K_shift)
        self.shift_cls_layer_f = nn.Linear(in_dim, 1 if self.use_binary_shift else self.K_shift_f)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_t=None, x_in_f=None):
        """
        Forward pass based on self.mode.

        Args:
            x_in_t: [B, C, L] time input
            x_in_f: [B, C, L] frequency input

        Returns:
            h_t, z_t, s_t, h_f, z_f, s_f, attn_w_t, attn_w_f
        """
        out_t = (None, None, None, None)
        out_f = (None, None, None, None)

        if self.mode in ['T', 'TF'] and x_in_t is not None:
            x_in_t = x_in_t.permute(0, 2, 1)  # [B, L, C]
            x_t, attn_maps_t = self.encoder_t(x_in_t.float())
            h_t = x_t.reshape(x_t.size(0), -1)
            z_t = self.projector_t(h_t)
            s_t = self.shift_cls_layer_t(h_t)
            if self.use_binary_shift:
                s_t = self.sigmoid(s_t)
            # Attention-weighted feature map per layer
            attn_w_t = [
                torch.bmm(attn.to(dtype=x_in_t.dtype, device=x_in_t.device), x_in_t)
                for attn in attn_maps_t
            ]
            out_t = (h_t, z_t, s_t, attn_w_t)

        if self.mode in ['F', 'TF'] and x_in_f is not None:
            x_in_f = x_in_f.permute(0, 2, 1)  # [B, L, C]
            x_f, attn_maps_f = self.encoder_f(x_in_f.float())
            h_f = x_f.reshape(x_f.size(0), -1)
            z_f = self.projector_f(h_f)
            s_f = self.shift_cls_layer_f(h_f)
            if self.use_binary_shift:
                s_f = self.sigmoid(s_f)
            attn_w_f = [
                torch.bmm(attn.to(dtype=x_in_f.dtype, device=x_in_f.device), x_in_f)
                for attn in attn_maps_f
            ]
            out_f = (h_f, z_f, s_f, attn_w_f)

        return (*out_t, *out_f)
