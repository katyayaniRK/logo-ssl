# LoGo-HTR: Self-Supervised Handwritten Text Recognition — Partial Replication
![Static Badge](https://img.shields.io/badge/Field-Deep%20Learning-blue) ![Static Badge](https://img.shields.io/badge/Domain-ComputerVision-green) ![Static Badge](https://img.shields.io/badge/Method-Self--SupervisedLearning-purple) ![Static Badge](https://img.shields.io/badge/Task-HandwrittenTextRecognition-orange) ![Static Badge](https://img.shields.io/badge/Platform-KaggleColab-yellow)

> Partial replication of the WACV 2026 paper **"Learning Beyond Labels: Self-Supervised Handwritten Text Recognition"** by Shree Mitra, Ajoy Mondal, and C.V. Jawahar (IIIT Hyderabad) — conducted on free-tier cloud platforms (Google Colab and Kaggle).

**What is LoGo-HTR?**

**LoGo-HTR (Local-Global HTR)** is a self-supervised learning framework for Handwritten Text Recognition (HTR) that reduces dependence on large labeled datasets by learning visual representations from unlabeled handwritten images.

It combines two complementary objectives:


- **Local Loss** — patch-wise contrastive loss that aligns spatial features across augmented views of the same image
- **Global Loss** — Barlow Twins decorrelation loss that reduces feature redundancy across embedding dimensions

### Two-Stage Pipeline
 
```
Stage 1: SSL Pretraining (LoGo-HTR)
─────────────────────────────────────────────────────────
  7.92M UNLABELED handwritten word images
        ↓
  DenseNet Encoder
  (Local contrastive loss + Global Barlow Twins loss)
        ↓
  Pretrained SSL Backbone
  (DenseNet_backbone_step290000.pth)
 
Stage 2: Supervised Fine-tuning
─────────────────────────────────────────────────────────
  Pretrained Encoder + CoMER Transformer Decoder
        ↓
  Fine-tune on labeled data
        ↓
  Word Recognition Rate (WRR) evaluation
```
 
---
## 📊 Results
 
### Our Replication Results (Progressive Experiments)
 
| Experiment | Platform | Training Samples | Epochs | Pretrained Backbone | val_WRR | val_loss |
|---|---|---|---|---|---|---|
| Experiment 1 | Google Colab | 300 | 10 | ❌ No | 16% | 2.173 |
| Experiment 2 | Google Colab | 300 | 10 | ✅ Yes | 17% | 2.204 |
| Experiment 3 | Google Colab | 2,000 | 10 | ✅ Yes | 20% | 1.685 |
| **Experiment 4** | **Kaggle** | **5,000** | **30** | **✅ Yes** | **45.5%** | **1.079** |
 
### Comparison with Paper
 
| Method | Training Samples | Epochs | WRR |
|---|---|---|---|
| Paper (Full) | 1.25M labeled | 100 | **87.93%** |
| Ours (Best) | 5,000 | 30 | **45.5%** |
 
> The gap is attributed to using only ~0.4% of the full training data and 30% of total epochs due to free-tier hardware constraints.
 
---
 
## 🗂️ Repository Structure
 
```
LoGoHTR/
├── train.py                          # Training entry point (modified for Lightning 2.x)
├── config.yaml                       # Training configuration
├── BarlowTwins.py                    # Barlow Twins loss implementation
├── Model.py                          # Model wrapper
├── DataModule.py                     # Data module wrapper
├── pre_training.py                   # SSL pretraining script
├── requirements.txt                  # Dependencies
├── comer/
│   ├── lit_comer.py                  # PyTorch Lightning module
│   ├── datamodule/
│   │   ├── datamodule.py             # Data loading and batching
│   │   ├── dataset.py                # Dataset class
│   │   ├── transforms.py             # Image augmentations
│   │   ├── vocab.py                  # Vocabulary (fixed: SSL_HWD_Dict.txt)
│   │   ├── SSL_HWD_Dict.txt          # SSL-HWD vocabulary
│   │   ├── IAM_Dict.txt              # IAM vocabulary
│   │   └── RIMES_Dictionary.txt      # RIMES vocabulary
│   ├── model/
│   │   ├── comer.py                  # CoMER model
│   │   ├── encoder.py                # DenseNet encoder (path fixed)
│   │   └── decoder.py                # Transformer decoder
│   ├── transformer/                  # Transformer blocks
│   └── utils/                        # Beam search and utilities
└── scripts/
    └── test/                         # Test scripts
```
 
---
 
## ⚙️ Setup and Installation
 
### Prerequisites
 
- Python 3.12
- GPU with at least 15GB VRAM (T4 or better)
- Google Colab / Kaggle (free tier) or local GPU machine
### Step 1 — Clone the Repository
 
```bash
# On Google Colab
import os
if not os.path.exists('/content/LoGoHTR'):
    !git clone https://github.com/logo-ssl/logo-ssl.git /content/LoGoHTR
%cd /content/LoGoHTR
 
# On Kaggle
import os
if not os.path.exists('/kaggle/working/LoGoHTR'):
    !git clone https://github.com/logo-ssl/logo-ssl.git /kaggle/working/LoGoHTR
%cd /kaggle/working/LoGoHTR
```
 
### Step 2 — Install Dependencies
 
```bash
pip install -q lightning einops editdistance \
    'jsonargparse[signatures]>=4.27.7' \
    opencv-python matplotlib typer beautifulsoup4 lxml
```
### Step 3 — Apply Compatibility Fixes
**Fix 1 — Replace pytorch_lightning imports (incompatible with Python 3.12):**
 
```python
import os
for root, dirs, files in os.walk('comer'):
    for file in files:
        if file.endswith('.py'):
            fpath = os.path.join(root, file)
            with open(fpath, 'r') as f:
                content = f.read()
            if 'pytorch_lightning' in content:
                content = content.replace('import pytorch_lightning',
                                          'import lightning.pytorch')
                content = content.replace('from pytorch_lightning',
                                          'from lightning.pytorch')
                with open(fpath, 'w') as f:
                    f.write(content)
print('Imports fixed!')
```
 
**Fix 2 — Fix missing dictionary file in vocab.py:**
 
```python
with open('comer/datamodule/vocab.py', 'r') as f:
    content = f.read()
content = content.replace('IIIT_Dict.txt', 'SSL_HWD_Dict.txt')
with open('comer/datamodule/vocab.py', 'w') as f:
    f.write(content)
print('vocab.py fixed!')
```
 
**Fix 3 — Rewrite train.py for Lightning 2.x:**
 
```python
new_train = """
import sys
sys.path.insert(0, '/content/LoGoHTR')  # update path for your environment
 
from lightning.pytorch.cli import LightningCLI
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
 
cli = LightningCLI(
    LitCoMER,
    CROHMEDatamodule,
    save_config_kwargs={'overwrite': True},
)
"""
with open('train.py', 'w') as f:
    f.write(new_train)
print('train.py fixed!')
```
 
---
## 🏋️ Training
 
### Step 1 — Download Pretrained SSL Backbone
 
```bash
wget -q \
  "https://huggingface.co/Shree10/LoGo-HTR/resolve/main/DenseNet_backbone_step290000.pth" \
  -O "/content/LoGoHTR/comer/model/BarlowTwins_HTR_backbone_step290000.pth"
```
 
### Step 2 — Fix Encoder Path
 
```python
with open('comer/model/encoder.py', 'r') as f:
    content = f.read()
content = content.replace(
    '/home2/shree.mitra23m/CoMER/comer/model/BarlowTwins_HTR_backbone_step290000.pth',
    '/content/LoGoHTR/comer/model/BarlowTwins_HTR_backbone_step290000.pth'
)
with open('comer/model/encoder.py', 'w') as f:
    f.write(content)
print('Encoder path fixed!')
```
 
### Step 3 — Configure Training
 
Update `config.yaml` with your dataset path:
 
```yaml
seed_everything: 7
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        monitor: val_WRR
        mode: max
        filename: '{epoch}-{step}-{val_WRR:.4f}'
        dirpath: /kaggle/working/checkpoints
  max_epochs: 30
  check_val_every_n_epoch: 2
  deterministic: true
  devices: 1
model:
  d_model: 256
  growth_rate: 24
  num_layers: 16
  nhead: 8
  num_decoder_layers: 3
  dim_feedforward: 1024
  dropout: 0.3
  dc: 32
  cross_coverage: true
  self_coverage: true
  beam_size: 10
  max_len: 200
  alpha: 1.0
  early_stopping: false
  temperature: 1.0
  learning_rate: 0.0008
  patience: 10
data:
  zipfile_path: /kaggle/working/data.zip   # update this path
  test_year: "2014"
  train_batch_size: 8
  eval_batch_size: 4
  num_workers: 4
  scale_aug: true
```
 
### Step 4 — Run Training
 
```bash
python train.py fit --config config.yaml
```
 
### Resume from Checkpoint
 
```bash
python train.py fit --config config.yaml \
  --ckpt_path /kaggle/working/checkpoints/last.ckpt
```
 
---
 
## 🖥️ Hardware Requirements
 
| Platform | GPU | GPU RAM | System RAM | Session Limit |
|---|---|---|---|---|
| Google Colab (Free) | T4 | 15GB | 12.7GB | ~12 hours |
| Kaggle (Free) | T4 x2 | 15GB each | 30GB | ~9 hours |
| Paper (Original) | 2x RTX A6000 | 48GB each | Large cluster | Days |
 
> **Recommendation**: Use **Kaggle** for more stable training sessions. Colab disconnects frequently on long runs.
 
---
 
## ⚠️ Known Issues and Fixes
 
| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pytorch_lightning` | Python 3.12 incompatibility | Replace with `lightning.pytorch` |
| `FileNotFoundError: IIIT_Dict.txt` | Missing dictionary file | Replace with `SSL_HWD_Dict.txt` |
| `FileNotFoundError: BarlowTwins_HTR_backbone` | Hardcoded author path | Download from HuggingFace, update path |
| `TypeError: save_config_overwrite` | Lightning 2.x API change | Use `save_config_kwargs={'overwrite': True}` |
| `ModuleNotFoundError: DDPPlugin` | Removed in Lightning 2.x | Remove DDP strategy, use `devices: 1` |
| `FileURLRetrievalError: gdown` | Google Drive quota limit | Use HuggingFace wget URL directly |
 
---


### ThankYou
