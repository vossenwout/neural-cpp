# Neural C++

A lightweight PyTorch-inspired deep learning framework implemented from scratch in C++. This is a hobby project to learn more about PyTorch and C++. 🤓

## Youtube series

I made a youtube series documenting my journey of building this library. Definitely subscribe if you are interested in AI 😀

- [Part 1: From PyTorch to C++](https://youtu.be/BTkIWUupQpk)
- [Part 2: Forward Pass](https://youtu.be/IJf3HhUJEeg)
- [Part 3: Automatic Differentiation](https://youtu.be/sqF4tAcOin0)
- [Part 4: Activation Functions and Modules](https://youtu.be/KTKd-EnLn6U)
- [Part 5: Loss Functions and Gradient Descent](https://youtu.be/s0P4xH14D1k)
- [Part 6: Training the Neural Network](https://youtu.be/yrCd-6s4E4A)

## Features

### 📊 **Tensors**

- **Multi-dimensional tensors** - Scalar, 1d and 2d tensor support.
- **Element-wise operations** - (+, \*, etc.)
- **Computational graph** - Automatically builds a computational graph.
- **Automatic differentiation** - Just call `tensor->backward()` to compute gradients

### 🧠 **Neural Network Modules**

- **Linear layers** - Fully connected layers with learnable weights and biases
- **Activation functions** - ReLU activation
- **Loss functions** - Cross-entropy loss for classification
- **Flatten** - Flatten layer for tensor reshaping

### 🚀 **Training**

- **SGD Optimizer** - Stochastic gradient descent for training.
- **Data Loading** - Easily batch your datasets
- **Model Serialization** - Save and load trained models
- **MNIST/FashionMNIST** - Implemented datasets classes

## Quick Start

### Prerequisites

- C++23 compatible compiler
- CMake 3.16+
- Git (for downloading GoogleTest)

### Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
cd build
./run_tests
```

### Training a Model

The repository includes a complete MNIST/FashionMNIST classification training example in `src/main.cpp`. To run it, you can use the following command after building the project:

```bash
cd .. # go back to the root directory else the dataset will not be found
./cpp_neural
```

The MNIST/FashionMNIST raw dataset files should be placed in
`data/MNIST/raw/` or `data/FashionMNIST/raw/`. If the files are not present,
download them from the official sources and place them in the corresponding
directories.

This example demonstrates:

- Loading MNIST/FashionMNIST datasets
- Training a multi-layer neural network
- Model evaluation
- Saving and loading trained models

## API Reference

### Core Components

- **`Tensor`** - Multi-dimensional arrays with automatic differentiation
- **`Module`** - Base class for all neural network components
- **`Linear`** - Fully connected layer
- **`Relu`** - ReLU activation function
- **`CrossEntropyLoss`** - Cross-entropy loss for classification
- **`SGD`** - Stochastic gradient descent optimizer
- **`DataLoader`** - Batch and shuffle your datasets

## Project Structure

```
neural_cpp/
├── include/nn/         # Header files
├── src/nn/             # Implementation files
├── tests/              # Test suite
├── models/             # Pre-trained models
├── data/               # Dataset storage
├── .clang-format       # Formatting configuration
└── CMakeLists.txt      # Build configuration
```

## Future extensions

- Support for accelerated compute (CUDA, MPS)
- Tensors with more dimensions (3d, 4d, etc.)
- More operations (Conv2d, MaxPool2d, etc.)
