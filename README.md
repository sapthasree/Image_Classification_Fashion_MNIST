# Fashion MNIST Image Classification Using Convolutional Neural Networks

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify grayscale images of clothing items from the Fashion MNIST dataset. The model is trained using PyTorch and achieves over **85% accuracy** on the validation dataset.

---

## Objective
- Build a CNN using PyTorch.
- Train and validate the model on the Fashion MNIST dataset.
- Visualize model performance with accuracy and cost (loss) plots.
- Display sample predictions from the validation dataset.

---

## Dataset Description
- **Dataset**: Fashion MNIST (from torchvision)
- **Image Size**: 28x28 pixels
- **Color**: Grayscale
- **Classes**: 10 categories including T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Environment Setup

### Required Libraries
List the necessary libraries used in this project:

- torch
- torchvision
- matplotlib
- numpy
- etc.

### Installation Instructions
Use pip or conda to install required packages.

```bash
# Example
pip install torch torchvision matplotlib
```

---
# Data Preprocessing
## Steps  
- Load the Fashion MNIST dataset using torchvision.datasets.  
- Normalize the pixel values to range [0, 1].  
- Convert the data into PyTorch DataLoader objects.  
- Split into training and validation sets.

---

# Model Architecture
## Layers Description
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Fully connected (Dense) layer
- Softmax output for classification
Use ``` torch.nn.Sequential ``` or a custom ```nn.Module``` class to define the architecture.
---

# Training the Model
## Key Components
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 10 (or more to improve accuracy)
- Batch Size: 64
- Device: CPU/GPU

## Training Process
- Loop through epochs
- Perform forward and backward pass
- Optimize weights
- Track training and validation accuracy/loss

---

# Validation and Evaluation
- Evaluate the model on the validation dataset.
- Ensure accuracy is above 85%.
- Print final validation accuracy.

---

# Visualizations
## 1. Sample Images from Validation Set
Display the first 3 validation images using a custom ```show_data()``` function.

![Screenshot 2025-06-11 122620](https://github.com/user-attachments/assets/f76358dc-f68d-4a58-b7d3-0fa067c5d860)


## 2. Accuracy and Loss Plots
Plot:
- Training vs Validation Accuracy
- Training vs Validation Loss

![Screenshot 2025-06-11 123943](https://github.com/user-attachments/assets/d85723f5-9c79-4931-8a1c-7a987ca22602)


Use ```matplotlib.pyplot``` for plotting.

---

# Results
- Final Validation Accuracy: Above 85%
- The model performs well in classifying unseen fashion items.
- Accuracy and loss trends show proper convergence.

---

# Certificate Highlight
🎓 Successfully completed "Deep Learning with PyTorch" course by IBM on Coursera with a 96.80% score.

