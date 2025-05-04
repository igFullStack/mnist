import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import argparse

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.layers(x)

def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished")

    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Training finished and model saved.")

def predict_digit(image_path):
    model = CNN()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or could not be opened.")

    img = cv2.resize(img, (28, 28))

    if np.mean(img) > 127:
        img = 255 - img

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5 

    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.predict:
        digit = predict_digit(args.predict)
        print(f"Predicted digit: {digit}")
    else:
        print("Specify --train to train or --predict <image_path> to predict.")
