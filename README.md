# Multilayer Perceptron (MLP) Neural Network - CSCI580 Project

This project is designed for the CSCI580 course. The task is to design and implement a **Multilayer Perceptron (MLP)** neural network using the **PyTorch** framework. No convolutional layers are required for this task. The goal is to demonstrate an understanding of neural network architecture and how to train a model for classification tasks using the MLP framework.

## Overview

In this project, we focus on:
- Building a simple MLP neural network from scratch using the PyTorch framework.
- Training the network on a dataset (in this case, the MNIST dataset of handwritten digits).
- Evaluating the performance of the model based on classification accuracy.

### Project Components
1. **Neural Network Architecture:**
   - The model architecture is a fully connected MLP with input, hidden, and output layers.
   - We use ReLU activation for the hidden layers and Softmax activation for the output layer to perform multi-class classification.

2. **Dataset:**
   - The MNIST dataset, containing images of handwritten digits (0-9), will be used for training and testing the model.

3. **Model Training:**
   - The model is trained using **Stochastic Gradient Descent (SGD)** with a momentum of 0.9.
   - We will use **Cross-Entropy Loss** as the loss function for multi-class classification.

4. **Evaluation:**
   - The model's performance is evaluated based on the accuracy of predictions on the MNIST test set.

## Requirements

To run the project, ensure you have the following Python packages installed:

- Python 3.x
- PyTorch
- torchvision
- numpy

Install the required dependencies using `pip`:

```bash
pip install torch torchvision numpy
