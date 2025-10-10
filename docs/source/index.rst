TCRfoundation Documentation
============================

**A multimodal foundation model for single-cell immune profiling**

TCRfoundation integrates gene expression and TCR sequences (α and β chains) from paired 
single-cell measurements through self-supervised pretraining with masked reconstruction 
and cross-modal contrastive learning.

.. image:: _static/overview1.png
   :width: 800
   :alt: TCRfoundation Overview

Features
--------

* Multimodal Learning: Integrates TCR sequences and gene expression
* Multiple Tasks: Classification, regression, and cross-modal prediction
* Pretrained Models: Ready-to-use foundation model
* Rich Analysis: Built-in evaluation and visualization tools

Quick Start
-----------

Installation::

    pip install tcrfoundation

Basic usage::

    import tcrfoundation as tcrf
    import scanpy as sc

    # Load pretrained model
    model = tcrf.load_foundation_model("path/to/checkpoint.pt")

    # Fine-tune for classification
    results, adata = tcrf.finetune.classification.train_classifier(
        adata, 
        label_column="cell_type",
        checkpoint_path="path/to/checkpoint.pt"
    )

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/01_pretrain.ipynb
   tutorials/02_classification.ipynb
   tutorials/03_specificity.ipynb
   tutorials/04_avidity.ipynb
   tutorials/05_cross_modal.ipynb


.. toctree::
   :maxdepth: 1
   :caption: About


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
