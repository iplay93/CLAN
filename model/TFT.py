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
