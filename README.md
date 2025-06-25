# simplified-cnn-fer
Facial Expression Recognition using PyTorch
This repository contains a PyTorch re-implementation of the Convolutional Neural Network models from the research paper "Facial Expression Recognition Using a Simplified Convolutional Neural Network Model" by Kandeel et al. (2021). The primary focus is the implementation and training of "Model 2" on the FER2013 dataset.

(Replace this with a screenshot of your prediction visualization output)

Table of Contents
Project Objective

Reference Paper

Model Architecture (Model 2)

Dataset

Getting Started

Prerequisites

Setup & Installation

Running the Code

Results

How to Use the Trained Model

Technologies Used

Project Objective
The goal of this project is to faithfully re-implement the deep learning models described in the reference paper and validate their performance on the FER2013 benchmark dataset. This includes setting up the data pipeline, defining the CNN architecture, and executing the training and evaluation process in a reproducible manner.

Reference Paper
Kandeel, A., Rahmanian, M., Zulkernine, F., Abbas, H. M., & Hassanein, H. (2021). Facial Expression Recognition Using a Simplified Convolutional Neural Network Model. 2020 International Conference on Communications, Signal Processing, and their Applications (ICCSPA).

IEEE Xplore Link

Model Architecture (Model 2)
This repository focuses on "Model 2," the more complex architecture designed for the challenging FER2013 dataset.

Input: 48x48 Grayscale Images

Convolutional Base:

Block 1: Conv(1, 64) -> ReLU -> Conv(64, 64) -> ReLU -> BatchNorm -> MaxPool

Block 2: Conv(64, 128) -> ReLU -> Conv(128, 128) -> ReLU -> BatchNorm -> MaxPool

Block 3: Conv(128, 256) -> ReLU -> Conv(256, 256) -> ReLU -> BatchNorm -> MaxPool

Classifier Head:

Flatten

Linear(9216, 256) -> ReLU -> Dropout(0.5)

Linear(256, 128) -> ReLU -> Dropout(0.5)

Linear(128, 7) (Output for 7 emotion classes)

Dataset
This project uses the FER2013 dataset, sourced from Kaggle.

Source: FER2013 on Kaggle

Structure:

Training set: 28,709 images

Test/Validation set: 7,178 images

Classes: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)

The code is designed to download and unzip this dataset automatically using the Kaggle API.

Getting Started
This project is designed to be run in a Google Colab environment to leverage its free GPU resources.

Prerequisites
A Google Account

A Kaggle Account

Setup & Installation
Clone the Repository (Optional):

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

Alternatively, you can create a new Colab notebook and copy the code from main.py or main.ipynb.

Get your Kaggle API Key:

Go to your Kaggle account page: https://www.kaggle.com/account

Scroll down to the "API" section.

Click "Create New API Token". This will download a kaggle.json file. Keep this file handy.

Running the Code
Open the main notebook in Google Colab.

Enable the GPU: In the Colab menu, go to Runtime -> Change runtime type and select GPU from the "Hardware accelerator" dropdown.

Run the first code cell. It will prompt you to upload the kaggle.json file you downloaded earlier.

The script will automatically:

Install the Kaggle library.

Configure your API key.

Download and unzip the FER2013 dataset.

Define the model and data transformations.

Train the model for 50 epochs, printing the loss and accuracy after each epoch.

Save the trained model weights as fer2013_model_2.pth.

Download the saved model file to your local computer.

Display a visualization of the model's predictions on a sample of test images.

Plot the training and validation history (loss and accuracy vs. epochs).

Results
After training for 50 epochs, the model achieved the following performance on the validation set:

Peak Validation Accuracy: 65.95%

Final Validation Loss: 1.1090

These results are in close alignment with the 69.32% accuracy reported in the reference paper, validating the re-implementation.

Training History
(Replace this with a screenshot of your generated plots)

How to Use the Trained Model
After running the script, the file fer2013_model_2.pth is downloaded to your computer. You can use this file to perform inference on new images without retraining.

import torch
from torchvision import transforms
from PIL import Image

# 1. Re-create the model architecture
# (Make sure the CnnModel2 class definition is available)
model = CnnModel2(num_classes=7)

# 2. Load the saved weights
model.load_state_dict(torch.load('fer2013_model_2.pth'))
model.eval() # Set the model to evaluation mode

# 3. Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 4. Load and process your image
img = Image.open("path/to/your/image.jpg")
img_tensor = transform(img).unsqueeze(0) # Add batch dimension

# 5. Make a prediction
with torch.no_grad():
    logits = model(img_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

# Map index to class name
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
predicted_emotion = class_names[predicted_class_index]

print(f"Predicted Emotion: {predicted_emotion}")

Technologies Used
Python

PyTorch

Pandas

NumPy

Matplotlib

Kaggle API

Google Colab
