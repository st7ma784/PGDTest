# PGD Attack on CLIP Embeddings with PyTorch Lightning

This repository investigates the use of Projected Gradient Descent (PGD) to attack CLIP embeddings. Additionally, we explore methods to generate clean images from these attacked embeddings.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
CLIP (Contrastive Language–Image Pre-training) is a neural network trained on a variety of (image, text) pairs. This project aims to:
1. Perform PGD attacks on CLIP embeddings.
2. Investigate techniques to reconstruct clean images from these embeddings.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/st7m784/PGDTest.git
cd PGDTest
pip install -r requirements.txt
```

## Usage
To run the PGD attack and investigate the reconstruction of clean images, use the following command:
```bash
python Launch.py
```

## Contributing
We welcome contributions! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.