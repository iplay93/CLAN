## 🧠🏃 CLAN: Self-supervised Dual-view Framework with Tailored Negative Sampling for New Activity Detection

With the rapid growth of ubiquitous computing, analyzing human activities from multi-sensor time series has become essential for enabling intelligent and context-aware services.

Most human activity recognition (HAR) models assume a fixed set of known activities. However, real-world scenarios often involve *new activity patterns* that were not observed during training. Detecting such novel activities at inference time remains a significant challenge due to:
1. Overlapping patterns between known and new activities,
2. Intra-class variability within the same activity, and
3. Heterogeneity in sensor modalities across datasets.

We present **CLAN**, a novel self-supervised learning framework based on **Contrastive Learning with multiple tailored data Augmentations for New activity detection**.

### 🔍 Key Features
- **Dual-tower architecture**  
  Captures multi-view discriminative features by combining time and frequency domain representations.
- **Robust contrastive learning**  
  Incorporates multiple types of *strongly shifted* augmented views as negatives, enabling the model to learn *invariant representations* that are robust to intra-class variations.
- **Dataset-specific augmentation selector**  
  Automatically tailors negative pair generation to each dataset’s characteristics, enhancing generalization across diverse sensor environments.

### 📊 Results
CLAN outperforms state-of-the-art baselines on several real-world sensor datasets, achieving up to **+9.24% improvement in AUROC**.
