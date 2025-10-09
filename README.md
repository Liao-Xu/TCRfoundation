# TCRfoundation

**A multimodal foundation model for single-cell immune profiling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

TCRfoundation integrates gene expression profiles and T cell receptor (TCR) sequences from paired single-cell measurements. The model uses self-supervised pretraining with masked reconstruction and cross-modal contrastive learning to enable comprehensive immune profiling across three key applications:

- **T-cell state classification**: Predict tissue origin, disease state, and cellular phenotype.
- **Binding specificity detection**: Identify TCR-antigen interactions and quantify binding avidity.
- **Cross-modal prediction**: Infer gene expression from TCR sequences.

![TCRfoundation Overview](docs/figures/overview.png)

## Installation

```bash
git clone https://github.com/Liao-Xu/TCRfoundation.git
cd TCRfoundation
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch 1.13.1+
