import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TFT(nn.Module):
    """
    TFT: Time-Frequency Transformer for Contrastive Learning with Shift Prediction.

    Supports:
    - Time-only mode ('T')
    - Frequency-only mode ('F')
    - Dual-view mode ('TF')
    - Contrastive projection and shift prediction (binary or multi-class)

    Returns:
        Depending on mode:
        h_time, z_time, s_time, h_freq, z_freq, s_freq
    """
    def __init__(self, configs, args):
        super(TFT, self).__init__()
        self.mode = args.training_mode  # 'T', 'F', or 'TF'
        self.use_binary_shift = getattr(args, "binary_shift", False)
        self.K_shift = getattr(args, "K_shift", 2)
        self.K_shift_f = getattr(args, "K_shift_f", 2)

        # --- Time Transformer ---
        self.encoder_t = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.TSlength_aligned,
                dim_feedforward=2 * configs.TSlength_aligned,
                nhead=1,
                batch_first=True
            ),
            num_layers=2
        )
        in_dim_t = configs.TSlength_aligned * configs.input_channels
        self.projector_t = nn.Sequential(
            nn.Linear(in_dim_t, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(in_dim_t, 1 if self.use_binary_shift else self.K_shift)

        # --- Frequency Transformer ---
        self.encoder_f = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.TSlength_aligned_2,
                dim_feedforward=2 * configs.TSlength_aligned_2,
                nhead=1,
                batch_first=True
            ),
            num_layers=2
        )
        in_dim_f = configs.TSlength_aligned_2 * configs.input_channels_2
        self.projector_f = nn.Sequential(
            nn.Linear(in_dim_f, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_f = nn.Linear(in_dim_f, 1 if self.use_binary_shift else self.K_shift_f)

        # Optional sigmoid for binary shift
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_t=None, x_in_f=None):
        """
        Forward pass depending on mode ('T', 'F', 'TF').

        Args:
            x_in_t: [B, L_t, C_t] time-domain input
            x_in_f: [B, L_f, C_f] frequency-domain input

        Returns:
            h_t, z_t, s_t, h_f, z_f, s_f (as applicable)
        """
        out_t = (None, None, None)
        out_f = (None, None, None)

        if self.mode in ['T', 'TF'] and x_in_t is not None:
            x_t = self.encoder_t(x_in_t.float())
            h_t = x_t.reshape(x_t.size(0), -1)
            z_t = self.projector_t(h_t)
            s_t = self.shift_cls_layer_t(h_t)
            if self.use_binary_shift:
                s_t = self.sigmoid(s_t)
            out_t = (h_t, z_t, s_t)

        if self.mode in ['F', 'TF'] and x_in_f is not None:
            x_f = self.encoder_f(x_in_f.float())
            h_f = x_f.reshape(x_f.size(0), -1)
            z_f = self.projector_f(h_f)
            s_f = self.shift_cls_layer_f(h_f)
            if self.use_binary_shift:
                s_f = self.sigmoid(s_f)
            out_f = (h_f, z_f, s_f)

        return (*out_t, *out_f)
