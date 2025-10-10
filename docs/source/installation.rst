Installation
============

Requirements
------------

* Python 3.8 or higher
* PyTorch 1.10 or higher
* CUDA (optional, for GPU support)

Install from PyPI
-----------------

The easiest way to install TCRfoundation::

    pip install tcrfoundation

Install from source
-------------------

For development or to get the latest features::

    git clone https://github.com/yourusername/TCRfoundation.git
    cd TCRfoundation
    pip install -e .

Verify installation
-------------------

Test that TCRfoundation is properly installed::

    python -c "import tcrfoundation as tcrf; print(tcrf.__version__)"

You should see the version number printed.
