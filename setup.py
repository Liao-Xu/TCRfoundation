from setuptools import setup, find_packages
import os

# Read version
version = {}
with open(os.path.join("tcrfoundation", "__version__.py")) as f:
    exec(f.read(), version)

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tcrfoundation",
    version=version["__version__"],
    author="Xu Liao",
    author_email="xl3514@cumc.columbia.edu",
    description="A multimodal foundation model for T cell receptor and transcriptome analysis",
    long_description_content_type="text/markdown",
    url="https://github.com/Liao-Xu/TCRfoundation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "jupyter>=1.0.0",
            "black>=21.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "furo",
            "myst-parser>=0.18.0",
            "myst-nb>=0.17.0",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "ipykernel",
            "nbformat",
        ],
    },
)
