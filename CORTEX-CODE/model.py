import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights and biases with a specific seed for reproducibility
np.random.seed(42)
input_size = 28 * 28  # Input layer size for MNIST images
hidden1_size = 32  # First hidden layer
hidden2_size = 16  # Second hidden layer
output_size = 10  # Output layer size (digits 0-9)

# Weights initialization
weights_input_hidden1 = np.random.randn(input_size, hidden1_size)
weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size)
weights_hidden2_output = np.random.randn(hidden2_size, output_size)

# Biases initialization
biases_hidden1 = np.random.randn(hidden1_size)
biases_hidden2 = np.random.randn(hidden2_size)
biases_output = np.random.randn(output_size)

def augment_image(image):
    # Reshape the image to its original 28x28 shape
    image = image.reshape((28, 28))
    
    # Rotate the image by a random angle within -30 to 30 degrees
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (28, 28))

    # Generate random noise
    randFloat = max(0.1, np.random.rand() * 0.25)
    noise = np.random.rand(28, 28) * randFloat
    noisy_image = rotated_image + noise

    # Blur the image
    blurred_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # Scale the image randomly
    scale = np.random.uniform(0.8, 1.2)  # Scale between 80% and 120%
    resized_image = cv2.resize(blurred_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Ensure resized_image has at least 28x28 dimensions
    if resized_image.shape[0] < 28 or resized_image.shape[1] < 28:
        resized_image = cv2.resize(resized_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Make sure the scaled image has shape (28, 28), fill the borders with black if necessary
    if resized_image.shape[0] > 28 or resized_image.shape[1] > 28:
        cropped_image = resized_image[int((resized_image.shape[0] - 28) / 2):int((resized_image.shape[0] + 28) / 2),
                                      int((resized_image.shape[1] - 28) / 2):int((resized_image.shape[1] + 28) / 2)]
    else:
        border_size_x = (28 - resized_image.shape[0]) // 2
        border_size_y = (28 - resized_image.shape[1]) // 2
        cropped_image = cv2.copyMakeBorder(resized_image, border_size_x, border_size_x, border_size_y, border_size_y, cv2.BORDER_CONSTANT, value=0)

    # Ensure cropped_image is 28x28
    assert cropped_image.shape == (28, 28), f"Augmented image is not of shape 28x28 after cropping/scaling, got {cropped_image.shape}."

    # Flatten the image back to 1D array
    augmented_image = cropped_image.flatten()

    # Ensure augmented_image is of the original flattened size
    assert augmented_image.shape == (input_size,), f"Augmented image is not of the correct flattened size, got {augmented_image.shape}."

    return augmented_image

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
                X_batch = np.array([augment_image(image) for image in X_batch])

            # Forward pass
            hidden1 = sigmoid(np.dot(X_batch, weights_input_hidden1) + biases_hidden1)
            hidden2 = sigmoid(np.dot(hidden1, weights_hidden1_hidden2) + biases_hidden2)
            output = sigmoid(np.dot(hidden2, weights_hidden2_output) + biases_output)

            # Calculate error
            error = y_batch - output

            # Backpropagation
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
        loss = np.mean(np.abs(error))

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
train(X_train, y_train, epochs=500, learning_rate=25e-5, batch_size=32, early_stopping_patience=10, graph=True, augment=True)
save_model('my_neural_network.npz')
# load_model('my_neural_network.npz')

