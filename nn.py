import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt

def load_data():
    if os.path.exists("train.csv"):
        data = pd.read_csv("train.csv").values
    else:
        data = np.empty((0, 785))
    return data

def preprocess_data(data):
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0  # Normalize pixels
    Y = data[:, 0]
    return X.T, Y

def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def relu_derivative(Z):
    return Z > 0

def one_hot(Y, classes=10):
    Y = Y.astype(int)
    one_hot_Y = np.zeros((classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def train(X, Y, alpha=0.1, iterations=500):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = np.mean(predictions == Y)
            print(f"Iteration {i}, Accuracy: {accuracy:.4f}")
    return W1, b1, W2, b2

def draw_digit():
    canvas = np.zeros((280, 280), dtype=np.uint8)
    drawing = False
    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(canvas, (x, y), 15, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow("Draw a digit")
    cv2.setMouseCallback("Draw a digit", draw)
    while True:
        cv2.imshow("Draw a digit", canvas)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break
    cv2.destroyAllWindows()
    return cv2.resize(canvas, (28, 28)).reshape(784)

def add_to_dataset(image, label):
    new_entry = np.hstack(([label], image))
    df = pd.DataFrame([new_entry])
    df.to_csv("train.csv", mode='a', header=not os.path.exists("train.csv"), index=False)

data = load_data()
X_train, Y_train = preprocess_data(data)
W1, b1, W2, b2 = train(X_train, Y_train)

while True:
    img = draw_digit()
    pred = np.argmax(forward_prop(W1, b1, W2, b2, img.reshape(-1, 1))[3], axis=0)
    print(f"Predicted: {pred[0]}")
    label = input("Enter actual label (or 'exit' to stop): ")
    if label.lower() == 'exit':
        break
    add_to_dataset(img, int(label))
