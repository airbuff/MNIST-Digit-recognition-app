# MNIST Digit Recognition 

## Overview
A PyTorch neural network application that recognizes handwritten digits using MNIST dataset and a tkinter GUI.

##Neural Network Architecture
Layer Breakdown

##Input Layer: 28x28 pixel flattened to 784 neurons
First Hidden Layer:

###784 → 128 neurons
ReLU activation


##Second Hidden Layer:

##128 → 64 neurons
ReLU activation


###Output Layer:

64 → 10 neurons (0-9 digit classification)



Training Parameters

Optimizer: Adam
Learning Rate: 0.001
Loss Function: Cross-Entropy
Epochs: 10
Batch Size: 64

Performance Metrics

Dataset: MNIST Handwritten Digits
Training Accuracy: Visualized during training
Model Type: Fully Connected Neural Network

Key Characteristics

Simple, interpretable architecture
Suitable for basic digit recognition
Relatively fast training
Moderate computational requirements

![image](https://github.com/user-attachments/assets/c5030688-36b9-4b90-8e34-1ecd88ca5808)


## Features
- Train or load pre-trained neural network model
- Draw digits in interactive GUI
- Real-time digit prediction
- Visualize training accuracy


![image](https://github.com/user-attachments/assets/3e7d8442-4b45-4ae3-ba83-674369a59a1c)

![image](https://github.com/user-attachments/assets/6c1c6213-89e4-4b84-9994-b614685ecc97)

![image](https://github.com/user-attachments/assets/9203ac5b-666e-485c-8d76-a65451127f5b)



## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- Pillow

## Installation
```bash
pip install torch torchvision pillow matplotlib
```

## How to Run
```bash
python digit_recognition.py
```
![image](https://github.com/user-attachments/assets/9c2c2e7d-9dd4-4d83-8f7b-10db5c4b9a7d)

![image](https://github.com/user-attachments/assets/1324c2c3-2638-48fd-98a3-f59e711c936b)

## Usage
1. Run the script
2. Draw a digit in the black canvas
3. Click "Predict" to see model's prediction
4. Use "Clear" to reset canvas

## Model Details
- Architecture: Fully Connected Neural Network
- Dataset: MNIST Handwritten Digits
- Training: 10 epochs with Adam optimizer

## Contributing
Pull requests welcome. For major changes, please open an issue first.

## License


[MIT](https://choosealicense.com/licenses/mit/)
