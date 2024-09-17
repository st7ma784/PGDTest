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

## Stephens Critique 

In this code, theres  lot that needs to be understood. Firstly, I figure It's worth looking at the oprimizers being used- for some reason there's  a little alrm bell ringing in my head that we should be using naAdam or Adam W optimizer over just SGD for training.... Might be worth consulting the originl paper ?

Secondly, I'm super curious about the janky way the prompts exist to create train classes, and generate prompts- especially just plainly repeating it across the batch rather than per class desired? (consult the demo notebook in the OpenAI repo?)

Thirdly, I am really rather uncertain about the direct training of the CLIP model. The pretrained models are pretty DAMN robust!!! any training, especially just on classes is going to undo that!? Otherwise, I really like the way they convert they do classification from a single encoder, it's very clever if it weren't for perhaps accidentally untraining the Visual Semantic Embeddings... (DEVISE would be a good exemplar paper to consult for this approach)