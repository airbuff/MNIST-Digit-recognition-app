import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import os

# Define the neural network
class DigitNN(nn.Module):
    def __init__(self):
        super(DigitNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

# Transform for both training and prediction
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data and create loader
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    model = DigitNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    
    accuracy_progress = []
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        accuracy_progress.append(accuracy)
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.4f}')
    
    plt.plot(range(1, epochs + 1), accuracy_progress)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Progress')
    plt.show()
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/digit_model.pth')
    return model

class DigitRecognitionApp:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.root = tk.Tk()
        self.root.title('Digit Recognition')
        
        # Canvas setup
        self.canvas = Canvas(self.root, width=200, height=200, bg='black')
        self.canvas.pack(pady=10)
        self.canvas.bind('<B1-Motion>', self.draw)
        
        # Create image for drawing
        self.canvas_image = Image.new('RGB', (200, 200), 'black')
        self.draw_img = ImageDraw.Draw(self.canvas_image)
        
        # Result label
        self.result_label = tk.Label(self.root, text='Draw a digit', font=('Arial', 14))
        self.result_label.pack(pady=10)
        
        # Buttons
        tk.Button(self.root, text='Predict', command=self.predict).pack(pady=5)
        tk.Button(self.root, text='Clear', command=self.clear_canvas).pack(pady=5)
    
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='white', outline='white')
        self.draw_img.ellipse([x-5, y-5, x+5, y+5], fill='white', outline='white')
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.canvas_image = Image.new('RGB', (200, 200), 'black')
        self.draw_img = ImageDraw.Draw(self.canvas_image)
        self.result_label.config(text='Draw a digit')
    
    def predict(self):
        # Convert the image for prediction
        img = self.canvas_image.convert('L')  # Convert to grayscale
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.model(img)
            prediction = torch.argmax(output, dim=1).item()
            self.result_label.config(text=f'Prediction: {prediction}')
    
    def run(self):
        self.root.mainloop()

def main():
    # Check if model exists, if not train a new one
    if os.path.exists('models/digit_model.pth'):
        model = DigitNN().to(device)
        model.load_state_dict(torch.load('models/digit_model.pth'))
        print("Loaded existing model")
    else:
        print("Training new model...")
        model = train_model()
    
    # Start the GUI application
    app = DigitRecognitionApp(model)
    app.run()

if __name__ == "__main__":
    main()