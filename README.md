# Deep Learning From Scratch — School Project Series

## Overview

This repository contains a series of deep learning projects implemented from scratch using only NumPy (and later PyTorch, when allowed), each corresponding to a progressive homework assignment. 
The goal was to understand and reimplement core components of modern machine learning models without relying on high-level libraries, gradually building up to more complex tasks.
Around 80% (100-95 on the early ones to 50-75 on the last ones) of the code is written by myself, with some inspiration to the current state of the art models and techniques.


---

## Project Structure

### **1. Multi-Layer Perceptron (MLP) — Homework 1**

> A fully functional feedforward neural network built entirely with NumPy.

**Features:**

* Fully customizable MLP architecture.
* Activation functions: ReLU, Sigmoid, Tanh.
* Loss functions: L2 Loss, Binary Cross-Entropy.
* Optimization methods:

  * Vanilla Gradient Descent
  * Momentum
  * ADAM
* Efficient backpropagation using caching of intermediate values.
* Dataset support: XOR, Swiss roll, and more.
* Visualization of decision boundaries.

**Example Usage:**

```python
from mlp import MLP
from optimizers import Adam
from loss import BCE
from activations import relu, sigmoid

model = MLP(input_dim=2, layers=[(2, 64), (64, 32), (32, 16), (16, 8), (8, 1)],
            activations=[relu, relu, relu, relu, sigmoid],
            loss_fn=BCE, init_method="xavier", optimizer=Adam())

DataTrain(epochs=500, learning_rate=0.01, model=model, dataset='swiss-roll')
```

---

### **2. Vision and CNNs — Homework 2**

> Extending from MLPs to visual data using convolutional architectures.

**Highlights:**

* Manual implementation of `Conv2D`, `MaxPool`, and related layers using NumPy.
* CNN trained on **CIFAR-100** from scratch.
* Implementation of:
  * Vanilla MLP trained on CIFAR-100.
  * Vanilla CNN
  * ResNet18 (residual blocks implemented manually)
* Run with:

```python
python mainCNN.py
```
* Multi-stage object detection using:

  * Sliding window + CNN classification
  * A Faster R-CNN–like pipeline with region proposals
* Comparison with YOLO-style detectors.
* Evaluation with **mean Average Precision (mAP)** computation.

---

### **3. Generative Adversarial Networks (GANs) — Homework 3**

> Generating data using adversarial training and exploring stability improvements.

**Core Contributions:**

* Implementation of standard GAN with:

  * Generator & Discriminator from scratch (NumPy then PyTorch).
  * Training loop with alternating updates.
* Advanced GAN variants:

  * Wasserstein GAN (with gradient penalty)
  * Least-Squares GAN
  * Spectral Normalization for stabilizing discriminator
* Evaluation on custom and standard image datasets.

---

### **4. Transformers and Language Models — Homework 4 (WIP)**

> Laying the foundation for understanding and building large language models.

**In Progress:**

* Manual implementation of key Transformer components:

  * Multi-head Self-Attention
  * Positional Encodings
  * Layer Normalization
* Planned:

  * Training a small LLM on toy datasets.
  * Exploring masked language modeling and text generation.

---

## Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>
pip install -r requirements.txt
```

## Dependencies

* NumPy
* (PyTorch for later parts, e.g., ResNet, GANs)
* Matplotlib (for visualization)
* tqdm

---

## Experiments and Visualizations

* Decision boundary plots (MLP and CNN classifiers).
* Training curves and loss landscapes.
* Generated samples from GANs.
* mAP curves for object detection.

---

## Author

Developed by Moussab — as part of a deep learning coursework exploring ML fundamentals by reimplementing key algorithms from scratch.
