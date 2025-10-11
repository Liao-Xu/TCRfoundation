# TCRfoundation

[![Documentation Status](https://readthedocs.org/projects/tcrfoundation/badge/?version=latest)](https://tcrfoundation.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/tcrfoundation.svg)](https://badge.fury.io/py/tcrfoundation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A multimodal foundation model for single-cell immune profiling**

## Overview

TCRfoundation integrates gene expression and TCR sequences (α and β chains) from paired single-cell measurements through self-supervised pretraining with masked reconstruction and cross-modal contrastive learning.

### Input and Pretraining Architecture

Gene expression profiles are encoded through feed-forward layers with multi-head attention, while TCR sequences are tokenized and processed through transformer blocks. The fused representations are learned via three objectives: masked gene expression reconstruction, masked TCR sequence reconstruction, and cross-modal alignment.

![Input and Pretraining](docs/figures/overview1.png)

### Fine-tuning Tasks

The pretrained model supports three downstream applications:

- **T-cell state classification**: Predict tissue origin, disease state, and cellular phenotype
- **Binding specificity detection**: Identify TCR-antigen interactions and quantify binding avidity
- **Cross-modal prediction**: Infer gene expression from TCR sequences

![Fine-tuning Tasks](docs/figures/overview2.png)

## Installation

### From PyPI (Recommended)

```bash
pip install tcrfoundation
```

### From Source

```bash
git clone https://github.com/Liao-Xu/TCRfoundation.git
cd TCRfoundation
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch 1.10.0+

## Quick Start

```python
import tcrfoundation as tcrf
import scanpy as sc

# Load your data
adata = sc.read("your_data.h5ad")

# Pretrain the foundation model
model, history = tcrf.pretrain.train(
    adata,
    epochs=500,
    batch_size=2048,
    save_dir='models/'
)

# Fine-tune for classification
results, adata_new = tcrf.finetune.classification.train_classifier(
    adata,
    label_column="cell_type",
    checkpoint_path="models/foundation_model_best.pt",
    num_epochs=50
)
```

## Documentation

- **Full Documentation**: [https://tcrfoundation.readthedocs.io](https://tcrfoundation.readthedocs.io)

## Tutorials

Complete Jupyter notebook tutorials are available:

1. [Pretraining](tutorials/01_pretrain.ipynb) - Train the foundation model
2. [Classification](tutorials/02_classification.ipynb) - T cell state classification
3. [Specificity](tutorials/03_specificity.ipynb) - Antigen specificity prediction
4. [Avidity](tutorials/04_avidity.ipynb) - Binding avidity regression
5. [Cross-modal](tutorials/05_cross_modal.ipynb) - TCR-to-gene prediction


## Contact

- **Author**: Xu Liao
- **Email**: xl3514@cumc.columbia.edu
- **GitHub**: [https://github.com/Liao-Xu/TCRfoundation](https://github.com/Liao-Xu/TCRfoundation)