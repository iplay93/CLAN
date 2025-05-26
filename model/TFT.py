import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TFT(nn.Module):
    """
    TFT: Time-Frequency Transformer for classification.
    
    - Two transformer encoders: one for time-domain, one for frequency-domain.
    - Project and fuse both features.
    - Final classifier head for downstream task.
    """
    def __init__(self, configs):
        super(TFT, self).__init__()

        # Time-domain Transformer
        self.encoder_t = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.TSlength_aligned,
                nhead=1,
                dim_feedforward=2 * configs.TSlength_aligned
            ),
            num_layers=2
        )

        # Frequency-domain Transformer
        self.encoder_f = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=configs.TSlength_aligned_2,
                nhead=1,
                dim_feedforward=2 * configs.TSlength_aligned_2
            ),
            num_layers=2
        )

        # Projection heads (if needed for dimension matching)
        self.proj_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 128),
            nn.ReLU()
        )

        self.proj_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned_2 * configs.input_channels_2, 128),
            nn.ReLU()
        )

        # Classifier: takes concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, configs.num_classes)
        )

    def forward(self, x_in_t, x_in_f):
        """
        Args:
            x_in_t: Time-domain input [B, L, C]
            x_in_f: Frequency-domain input [B, L_f, C_f]

        Returns:
            logits: [B, num_classes]
        """
        # Time transformer
        t_feat = self.encoder_t(x_in_t.float())
        t_feat_flat = t_feat.reshape(t_feat.size(0), -1)
        t_proj = self.proj_t(t_feat_flat)

        # Frequency transformer
        f_feat = self.encoder_f(x_in_f.float())
        f_feat_flat = f_feat.reshape(f_feat.size(0), -1)
        f_proj = self.proj_f(f_feat_flat)

        # Fuse and classify
        fused = torch.cat([t_proj, f_proj], dim=1)
        logits = self.classifier(fused)
        return logits


class TFT_tas(nn.Module):
    """
    TFT_tas: Dual-view Transformer for contrastive learning with shift prediction.
    - One encoder for time-domain
    - One encoder for frequency-domain
    - Each path outputs latent, projected, and shift-predicted representations
    """
    def __init__(self, configs, args):
        super(TFT_tas, self).__init__()
        self.training_mode = args.training_mode

        # Time-domain Transformer
        encoder_layer_t = TransformerEncoderLayer(
            d_model=configs.TSlength_aligned,
            dim_feedforward=2 * configs.TSlength_aligned,
            nhead=1,
            batch_first=True
        )
        self.encoder_t = TransformerEncoder(encoder_layer_t, num_layers=2)

        # Frequency-domain Transformer
        encoder_layer_f = TransformerEncoderLayer(
            d_model=configs.TSlength_aligned_2,
            dim_feedforward=2 * configs.TSlength_aligned_2,
            nhead=1,
            batch_first=True
        )
        self.encoder_f = TransformerEncoder(encoder_layer_f, num_layers=2)

        # Projectors
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned_2 * configs.input_channels_2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Shift classification heads (used in contrastive or self-supervised learning)
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, 1)
        self.shift_cls_layer_f = nn.Linear(configs.TSlength_aligned_2 * configs.input_channels_2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_t, x_in_f):
        """
        Args:
            x_in_t: [B, L_t, C_t] time-domain input
            x_in_f: [B, L_f, C_f] frequency-domain input

        Returns:
            h_time: raw time-domain features
            z_time: projected time-domain features
            s_time: shift score
            h_freq, z_freq, s_freq: same for frequency-domain
        """
        # Time domain path
        t_feat = self.encoder_t(x_in_t.float())                       # [B, L, C]
        h_time = t_feat.reshape(t_feat.size(0), -1)                  # [B, L*C]
        z_time = self.projector_t(h_time)                            # [B, 128]
        s_time = self.sigmoid(self.shift_cls_layer_t(h_time))        # [B, 1]

        # Frequency domain path
        f_feat = self.encoder_f(x_in_f.float())                      # [B, L_f, C_f]
        h_freq = f_feat.reshape(f_feat.size(0), -1)                  # [B, L*C]
        z_freq = self.projector_f(h_freq)                            # [B, 128]
        s_freq = self.sigmoid(self.shift_cls_layer_f(h_freq))        # [B, 1]

        return h_time, z_time, s_time, h_freq, z_freq, s_freq