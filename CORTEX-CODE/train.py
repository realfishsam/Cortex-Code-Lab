import os
import numpy as np
import cv2
from neuralnet import NeuralNetwork  # Importing the NeuralNetwork class

# Initialize the neural network
nn = NeuralNetwork()

# Set a specific seed for reproducibility
np.random.seed(42)

# Define sizes for each layer
input_size = 28 * 28  # Input layer size for MNIST images
hidden1_size = 32     # First hidden layer
hidden2_size = 32     # Second hidden layer
output_size = 10      # Output layer size (digits 0-9)

# Initialize weights and biases
nn.weights_input_hidden1 = np.random.randn(input_size, hidden1_size)
nn.weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size)
nn.weights_hidden2_output = np.random.randn(hidden2_size, output_size)

nn.biases_hidden1 = np.random.randn(hidden1_size)
nn.biases_hidden2 = np.random.randn(hidden2_size)
nn.biases_output = np.random.randn(output_size)

# Load the dataset
dataset_directory = 'trainingSet/trainingSet'
X, y = nn.load_data(dataset_directory)

# One-hot encode the labels
num_classes = 10  # Because you have 10 classes (digits 0-9)
y = nn.one_hot_encode(y, num_classes)

# Print the shapes to confirm the loading process
print("Shape of X:", X.shape)  # Should be (number_of_samples, 784)
print("Shape of y:", y.shape)  # Should be (number_of_samples, num_classes)

augment_settings = {
    'scale_range': (0.8, 1.2),
    'angle_range': (-30, 30),
}

# Train the neural network
nn.train(X, y, epochs=500, learning_rate=8e-4, batch_size=32, early_stopping_patience=10, graph=True, augment_settings=augment_settings)

# Save the trained model
nn.save_model('my_neural_network.npz')

# Uncomment the following line if you want to load a saved model
# nn.load_model('my_neural_network.npz')
