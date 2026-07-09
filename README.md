# Handwritten Digit Recognition using a Neural Network

## Overview

This project implements a **Feedforward Neural Network (Multi-Layer Perceptron)** from scratch using **Python** and **NumPy** to recognize handwritten digits from the MNIST dataset. The objective was to understand the core concepts behind neural networks by implementing every component manually, without relying on deep learning frameworks such as TensorFlow or PyTorch.

The model is trained using forward propagation, backpropagation, gradient descent, ReLU activation, and Softmax activation to classify handwritten digits (0–9).

---

## Features

* Built entirely from scratch using Python and NumPy
* Uses the MNIST handwritten digit dataset
* Implements:

  * Forward Propagation
  * Backpropagation
  * Gradient Descent
  * ReLU Activation
  * Softmax Activation
* Predicts handwritten digits from 0 to 9
* Visualizes predictions on test images

---

## Model Architecture

```text
Input Layer
784 Neurons (28 × 28 pixels)

        ↓

Hidden Layer
10 Neurons
ReLU Activation

        ↓

Output Layer
10 Neurons
Softmax Activation
```

---

## Dataset

* **Training Samples:** 4,000
* **Validation Samples:** 1,000
* **Image Size:** 28 × 28 pixels
* **Classes:** 10 (Digits 0–9)

---

## Training Details

* Optimizer: Gradient Descent
* Activation Function: ReLU
* Output Activation: Softmax
* Training Iterations: **500**
* Loss Function: Cross-Entropy Loss

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## Project Structure

```text
MNIST-Digit-Recognition/
│
├── data/
│   └── mnist.csv
│
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

## Results

The neural network successfully learns to recognize handwritten digits by updating its weights through backpropagation over multiple training iterations. After training, the model is capable of predicting unseen handwritten digits with high accuracy.

---

## Future Improvements

* Implement a Convolutional Neural Network (CNN)
* Train on the complete MNIST dataset
* Improve accuracy using deeper architectures
* Build a web interface using Gradio or Streamlit
* Deploy the model online for real-time digit recognition

---

## Learning Outcomes

This project helped me understand:

* Neural Network fundamentals
* Matrix operations in NumPy
* Forward and Backpropagation
* Gradient Descent optimization
* ReLU and Softmax activations
* Model training and evaluation
* Image classification workflow

---

## How to Run

1. Clone the repository.
2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook.

```bash
jupyter notebook
```

4. Run all cells to train the model and make predictions.

---

## Author

**Ayush Bisht**

B.Tech Computer Science & Engineering
Passionate about Machine Learning, Artificial Intelligence, and Full-Stack Development.
