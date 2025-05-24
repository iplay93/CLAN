# 🧠🏃 CLAN: Self-supervised Dual-view Framework with Tailored Negative Sampling for New Activity Detection

CLAN is a **self-supervised dual-view framework** designed for robust **new activity detection** in human activity recognition (HAR) systems. It learns discriminative and invariant representations from **only known activity data**, without requiring access to any novel activity samples during training.

![CLAN Overview](overview.png)
#### Figure: CLAN framework. It consists of tailored Data Augmentation Set Construction, Discriminative Representation Learning, and Novelty Detection stages.


## 🚀 Motivation

Most HAR systems assume a fixed set of known activities. However, in real-world scenarios, *new activity patterns*—unseen during training—frequently occur. Detecting such activities at inference time is challenging due to:

- Overlapping patterns between known and new activities  
- High intra-class variability within known activities  
- Heterogeneous sensor modalities and dataset domains  

## 🔧 Core Components

#### 1. Multi-view Decomposition
To reduce feature entanglement and capture complementary information, CLAN uses a **two-tower encoder**:
- **Time-domain encoder** captures local temporal structures
- **Frequency-domain encoder** captures periodic and spectral patterns

#### 2. Diverse Augmentation Repulsion Learning
To ensure robustness to intra-class variations, CLAN:
- Applies **multiple strong augmentations** to each known activity sample
- Treats the augmented views as **negatives** in contrastive learning  
→ Encourages learning of **transformation-invariant** representations

#### 3. Dataset-aware Augmentation Selection
To adapt to diverse sensor environments, CLAN:
- Trains a simple **binary classifier** to evaluate augmentation strength
- Automatically **selects effective augmentation strategies** per dataset

---

## 📚 Benchmark Datasets

We evaluate CLAN on five heterogeneous human activity recognition (HAR) datasets, covering diverse sensor modalities, environments, and activity types:

- **OPPORTUNITY (OPP)**  
  Daily activities (e.g., *coffee time*) recorded in a controlled indoor setting with **242 wearable and ambient sensors**, including 3D accelerometers in drawers.  
  [[📎 Dataset]](https://archive.ics.uci.edu/dataset/226/opportunity%2Bactivity%2Brecognition)

- **CASAS**  
  Multi-user activities (e.g., *packing a picnic basket*) captured in a smart home equipped with **51 motion**, **8 door**, and **2 item sensors**.  
  [[📎 Dataset]](http://casas.wsu.edu/datasets/)

- **ARAS**  
  Simple daily tasks (e.g., *watching TV*) performed by multiple residents in a smart home with **20 types of ambient sensors**.   
  [[📎 Dataset]](https://github.com/SeyedHamidreza/ARAS-dataset)

- **DOO-RE**  
  Group activities (e.g., *seminar*) recorded in a university seminar room using **7 ambient sensor types**, such as seat occupancy and light sensors.  
  [[📎 Dataset]](https://github.com/dasom-elab/DOO-RE)

- **OPENPACK (OPEN)**  
  Industrial tasks (e.g., *relocating product labels*) collected using **4 IMUs**, **2 biosignal sensors**, and **2 custom devices**, sampled at ~30 Hz.   
  [[📎 Dataset]](https://open-pack.github.io/)



## 📊 Results

CLAN consistently outperforms state-of-the-art baselines on five real-world HAR benchmarks, achieving up to **+9.24% improvement in AUROC** for detecting new activities.

---

