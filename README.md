# Handwritten Digit Recognition using Convolutional Neural Network (CNN)

## Overview

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The CNN is built using TensorFlow and Keras, and it is trained on the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9).

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Matplotlib

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/atharv-451/Digit-Recognition-using-CNN.git
cd Digit-Recognition-using-CNN
```

Install the required Dependencies
```bash
pip install tensorflow matplotlib
```

## Usage
1. Open the Jupyter notebook or Python script containing the code.
2. Run each cell or execute the script to load the MNIST dataset, preprocess the data, build the CNN model, and train the model.
3. After training, the model will be evaluated on the testing dataset, and predictions will be displayed.

## Code Structure
- 'Hand-Written Digit Recognition using MNIST Dataset.ipynb': Jupyter notebook containing the main code.
- 'Hand-Written Digit Recognition using MNIST Dataset.py': Python script equivalent to the notebook

## Dataset

The MNIST dataset is used for training and testing the model. It is loaded using TensorFlow's built-in dataset loading functionality.

## Model Architecture

The CNN model consists of three convolutional layers with ReLU activation functions, followed by max-pooling layers. After the convolutional layers, there are two fully connected layers with ReLU activation functions. The final layer is a fully connected layer with 10 output nodes, representing digits 0-9, and it uses the softmax activation function.

## Training

The model is trained on 60,000 samples from the MNIST training dataset for 5 epochs. The training process includes normalization of the input data and resizing of images to prepare them for convolutional operations.

## Evaluation

After training, the model is evaluated on the MNIST testing dataset, consisting of 10,000 samples. The test loss and accuracy are reported.

## Results

The trained model can make predictions on new handwritten digit images, and the predictions are displayed for visual inspection.

## Acknowledgments

- The MNIST dataset is a product of Yann LeCun and Corinna Cortes at NYU.
- The code in this repository is inspired by various examples and tutorials available in the TensorFlow documentation.

### Feel free to contribute to the project by opening issues or creating pull requests. Happy coding!
