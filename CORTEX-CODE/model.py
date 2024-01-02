import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def cross_entropy(y_pred, y_true):
    # y_pred - predictions from neural network, y_true - one hot encoded true labels
    m = y_true.shape[0]
    loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-15))  # Adding a small value to avoid log(0)
    return loss

def conv2d(X, W, b, stride=1, padding=0):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width W.
    We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

    Input:
    - X: Input data of shape (N, C, H, W)
    - W: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - stride: The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - padding: The number of pixels that will be used to zero-pad the input.
    
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * padding - HH) / stride
      W' = 1 + (W + 2 * padding - WW) / stride
    """
    N, C, H, W = X.shape
    F, _, HH, WW = W.shape
    H_out = 1 + (H + 2 * padding - HH) // stride
    W_out = 1 + (W + 2 * padding - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    # Apply padding to the input
    X_padded = np.pad(X, ((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for height in range(0, H_out):
                for width in range(0, W_out):
                    h_start = height * stride
                    h_end = h_start + HH
                    w_start = width * stride
                    w_end = w_start + WW
                    
                    window = X_padded[n, :, h_start:h_end, w_start:w_end]
                    
                    out[n, f, height, width] = np.sum(window * W[f, :, :, :]) + b[f]

    return out

# Initialize weights and biases with a specific seed for reproducibility
np.random.seed(42)
input_size = 28 * 28  # Input layer size for MNIST images
hidden1_size = 32  # First hidden layer
hidden2_size = 32  # Second hidden layer
output_size = 10  # Output layer size (digits 0-9)

# Weights initialization
weights_input_hidden1 = np.random.randn(input_size, hidden1_size)
weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size)
weights_hidden2_output = np.random.randn(hidden2_size, output_size)

# Biases initialization
biases_hidden1 = np.random.randn(hidden1_size)
biases_hidden2 = np.random.randn(hidden2_size)
biases_output = np.random.randn(output_size)

def augment_image(image, scale_range=(1.0, 1.0), angle_range=(0, 0), translation_range=(0, 0), white_noise_percentage=0.0):
    input_size = 28 * 28  # Since the image is 28x28

    # Reshape the image to its original 28x28 shape
    image = image.reshape((28, 28))
    
    # Choose a random scale, angle, and translation from the given ranges
    scale_level = np.random.uniform(scale_range[0], scale_range[1])
    angle_level = np.random.uniform(angle_range[0], angle_range[1])
    tx = np.random.randint(translation_range[0], translation_range[1] + 1)
    ty = np.random.randint(translation_range[0], translation_range[1] + 1)
    
    # Apply the rotation and scale transformations
    M = cv2.getRotationMatrix2D((14, 14), angle_level, scale_level)
    image = cv2.warpAffine(image, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Apply the translation transformation
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_trans, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Add random white noise based on the percentage specified
    if white_noise_percentage > 0:
        num_pixels = int(input_size * white_noise_percentage)
        indices = np.random.choice(range(input_size), num_pixels, replace=False)
        flat_image = image.flatten()
        flat_image[indices] = 255  # Set chosen pixels to white
        image = flat_image.reshape((28, 28))

    # return image.flatten(), scale_level, angle_level, (tx, ty)
    return image.flatten()

# Training the neural network
def train(X, y, epochs, learning_rate, batch_size, early_stopping_patience, graph=False, augment=False):
    global weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output
    global biases_hidden1, biases_hidden2, biases_output

    loss_history = []
    accuracy_history = []

    best_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, X.shape[0], batch_size):
            # Mini-batch data
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Apply data augmentation if enabled
            if augment:
                X_batch = np.array([augment_image(
                    image,
                    scale_range=(0.8, 1.2),
                    angle_range=(-30, 30),
                    # translation_range=(-5, 5),
                    # white_noise_percentage=0.1
                    ) for image in X_batch])

            # Forward pass with ReLU
            hidden1 = sigmoid(np.dot(X_batch, weights_input_hidden1) + biases_hidden1)
            hidden2 = sigmoid(np.dot(hidden1, weights_hidden1_hidden2) + biases_hidden2)
            output = sigmoid(np.dot(hidden2, weights_hidden2_output) + biases_output)


            # Backpropagation with ReLU derivatives
            error = y_batch - output
            d_output = error * sigmoid_derivative(output)
            d_hidden2 = np.dot(d_output, weights_hidden2_output.T) * sigmoid_derivative(hidden2)
            d_hidden1 = np.dot(d_hidden2, weights_hidden1_hidden2.T) * sigmoid_derivative(hidden1)

            # Update weights and biases
            weights_hidden2_output += np.dot(hidden2.T, d_output) * learning_rate
            biases_output += np.sum(d_output, axis=0) * learning_rate
            weights_hidden1_hidden2 += np.dot(hidden1.T, d_hidden2) * learning_rate
            biases_hidden2 += np.sum(d_hidden2, axis=0) * learning_rate
            weights_input_hidden1 += np.dot(X_batch.T, d_hidden1) * learning_rate
            biases_hidden1 += np.sum(d_hidden1, axis=0) * learning_rate

        # Calculate loss and accuracy for the entire dataset for reporting
        hidden1 = sigmoid(np.dot(X, weights_input_hidden1) + biases_hidden1)
        hidden2 = sigmoid(np.dot(hidden1, weights_hidden1_hidden2) + biases_hidden2)
        output = sigmoid(np.dot(hidden2, weights_hidden2_output) + biases_output)
        error = y - output
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        loss = cross_entropy(output, y)

        # Save the loss and accuracy in their respective history lists
        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        print(f'Epoch {epoch}\tLoss: {loss}\tAccuracy: {accuracy * 100}%')

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break


    if graph:
        import matplotlib.pyplot as plt
        plt.plot(loss_history, label='Loss')
        plt.plot(accuracy_history, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'loss_and_accuracy{learning_rate}.png')

# Function to save the trained model parameters
def save_model(file_path):
    np.savez(file_path,
             weights_input_hidden1=weights_input_hidden1,
             biases_hidden1=biases_hidden1,
             weights_hidden1_hidden2=weights_hidden1_hidden2,
             biases_hidden2=biases_hidden2,
             weights_hidden2_output=weights_hidden2_output,
             biases_output=biases_output)

# Function to load the trained model parameters
def load_model(file_path):
    data = np.load(file_path)
    global weights_input_hidden1, biases_hidden1, weights_hidden1_hidden2, biases_hidden2, weights_hidden2_output, biases_output
    weights_input_hidden1 = data['weights_input_hidden1']
    biases_hidden1 = data['biases_hidden1']
    weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
    biases_hidden2 = data['biases_hidden2']
    weights_hidden2_output = data['weights_hidden2_output']
    biases_output = data['biases_output']


import os
import numpy as np
import cv2


# Function to load images and labels from given directory
def load_data(directory):
    images = []
    labels = []
    for label_folder in os.listdir(directory):
        label_folder_path = os.path.join(directory, label_folder)
        if os.path.isdir(label_folder_path):
            for image_filename in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_filename)
                # Read the image in grayscale mode
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Flatten the image to a 1D array of 28*28 = 784 pixels
                    image_flattened = image.flatten()
                    images.append(image_flattened)
                    labels.append(int(label_folder))  # Label is the folder name
    return np.array(images), np.array(labels)

# Define the directory path where the MNIST dataset folders are located
dataset_directory = 'trainingSet/trainingSet'

# Load the dataset
X, y = load_data(dataset_directory)

def one_hot_encode(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int_)

    # Initialize matrix of zeros
    one_hot = np.zeros((labels.size, num_classes))
    
    # np.arange(labels.size) creates an array with indices from 0 to the length of labels - 1
    # labels.reshape(-1) ensures that labels is a proper 1D array
    # This will then index the one_hot array and set the appropriate element to 1
    one_hot[np.arange(labels.size), labels.reshape(-1)] = 1
    return one_hot

num_classes = 10  # Because you have 10 classes (digits 0-9)
# y_one_hot = one_hot_encode(y, num_classes)
y = one_hot_encode(y, num_classes)

# Assuming your labels are in the variable `y`:
num_classes = 10  # Because you have 10 classes (digits 0-9)
y_one_hot = one_hot_encode(y, num_classes)


# Print the shapes to confirm the loading process
print("Shape of X:", X.shape)  # Should be (number_of_samples, 784)
print("Shape of y:", y.shape)  # Should be (number_of_samples,)

X_train, y_train = X,y
# Example: Uncomment the following lines to train and save your model
train(X_train, y_train, epochs=500, learning_rate=8e-4, batch_size=32, early_stopping_patience=10, graph=True, augment=True)
save_model('my_neural_network.npz')
# load_model('my_neural_network.npz')