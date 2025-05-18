## 🧠 CLAN: Contrastive Learning with Tailored Augmentations for Novel Activity Detection

With the growing adoption of ubiquitous computing, analyzing human activities from multi-sensor time series has become essential for enabling intelligent and context-aware services.

Most human activity recognition (HAR) models assume a fixed set of known activities; however, real-world scenarios often involve *new activity patterns* that are not observed during training. Detecting such novel activities at inference time remains a significant challenge due to:
1. Overlapping patterns between known and new activities,
2. Intra-class variability within the same activity, and
3. Heterogeneity in sensor modalities across datasets.

We propose **CLAN**, a novel self-supervised learning framework based on **Contrastive Learning with multiple tailored data Augmentations for Novel activity detection**.

### 🔍 Key Features
- **Dual-tower architecture**: Leverages both time and frequency domain representations to learn multi-view discriminative features.
- **Robust contrastive learning**: Uses multiple types of *strongly shifted* augmented views as negatives, encouraging the model to learn *invariant representations* that are robust to intra-class variations.
- **Dataset-specific augmentation selector**: Automatically tailors negative pair generation based on the characteristics of each dataset, improving generalization across diverse sensor environments.

### 📊 Results
CLAN significantly outperforms state-of-the-art baselines on multiple real-world sensor datasets, achieving up to **+9.24% improvement in AUROC**.
