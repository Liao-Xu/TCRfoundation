# TCRfoundation

[![Documentation Status](https://readthedocs.org/projects/tcrfoundation/badge/?version=latest)](https://tcrfoundation.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A multimodal foundation model for single-cell immune profiling**

**A multimodal foundation model for single-cell immune profiling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
```bash
git clone https://github.com/Liao-Xu/TCRfoundation.git
cd TCRfoundation
pip install -e .

**Requirements**: Python 3.8+, PyTorch 1.13.1+