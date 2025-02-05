# MNIST Digit Recognition 

## Overview
A PyTorch neural network application that recognizes handwritten digits using MNIST dataset and a tkinter GUI.

## Features
- Train or load pre-trained neural network model
- Draw digits in interactive GUI
- Real-time digit prediction
- Visualize training accuracy

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