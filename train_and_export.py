import numpy as np
import pandas as pd
import json
import os

def load_data():
    print("Loading train.csv...")
    if not os.path.exists("train.csv"):
        # If running from inside Digit Recognition and train.csv is in parent dir, search there
        if os.path.exists("../train.csv"):
            data = pd.read_csv("../train.csv")
        else:
            raise FileNotFoundError("Could not find train.csv in current or parent directory.")
    else:
        data = pd.read_csv("train.csv")
        
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    # Dev set (1000 samples)
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    
    # Training set
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    
    return X_train, Y_train, X_dev, Y_dev

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m_samples = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_samples * dZ2.dot(A1.T)
    db2 = 1 / m_samples * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_samples * dZ1.dot(X.T)
    db1 = 1 / m_samples * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i:3d}: Training Accuracy = {acc * 100:.2f}%")
    return W1, b1, W2, b2

def main():
    X_train, Y_train, X_dev, Y_dev = load_data()
    print(f"Dataset loaded. Train shape: {X_train.shape}, Dev shape: {X_dev.shape}")
    
    print("Training model...")
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 500)
    
    # Validation evaluation
    _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
    dev_preds = get_predictions(A2_dev)
    dev_acc = get_accuracy(dev_preds, Y_dev)
    print(f"Validation Set Accuracy: {dev_acc * 100:.2f}%")
    
    # Save parameters to JSON
    weights_path = "mnist_weights.json"
    print(f"Saving weights to {weights_path}...")
    weights_data = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist()
    }
    
    with open(weights_path, "w") as f:
        json.dump(weights_data, f)
    
    print("Weights exported successfully!")

if __name__ == "__main__":
    main()
