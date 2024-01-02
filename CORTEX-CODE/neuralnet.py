import numpy as np
import cv2
import os

class NeuralNetwork:
    """
    This class implements a simple feedforward neural network for image classification.

    Attributes:
        weights_input_hidden1 (numpy.ndarray): Weights matrix for input to first hidden layer.
        biases_hidden1 (numpy.ndarray): Bias vector for the first hidden layer.
        weights_hidden1_hidden2 (numpy.ndarray): Weights matrix for first to second hidden layer.
        biases_hidden2 (numpy.ndarray): Bias vector for the second hidden layer.
        weights_hidden2_output (numpy.ndarray): Weights matrix for second hidden layer to output.
        biases_output (numpy.ndarray): Bias vector for the output layer.
    """
    
    def __init__(self):
        """
        Initialize the neural network with all weights and biases set to None.
        These weights and biases are for three layers: input to hidden layer 1, 
        hidden layer 1 to hidden layer 2, and hidden layer 2 to output layer.
        """
        self.weights_input_hidden1 = None
        self.biases_hidden1 = None
        self.weights_hidden1_hidden2 = None
        self.biases_hidden2 = None
        self.weights_hidden2_output = None
        self.biases_output = None

    def relu(self, x):
        """
        Apply the ReLU activation function.

        Args:
            x (numpy.ndarray): Input array or matrix.

        Returns:
            numpy.ndarray: ReLU applied output.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Calculate the derivative of the ReLU function.

        Args:
            x (numpy.ndarray): Input array or matrix.

        Returns:
            numpy.ndarray: Derivative of ReLU.
        """
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """
        Apply the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array or matrix.

        Returns:
            numpy.ndarray: Sigmoid function output.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Calculate the derivative of the sigmoid function.

        Args:
            x (numpy.ndarray): Input array or matrix.

        Returns:
            numpy.ndarray: Derivative of sigmoid.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cross_entropy(self, y_pred, y_true):
        """
        Compute the cross-entropy loss.

        Args:
            y_pred (numpy.ndarray): Predicted output probabilities.
            y_true (numpy.ndarray): True labels.

        Returns:
            float: Cross entropy loss.
        """
        m = y_true.shape[0]
        loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-15))
        return loss

    def augment_image(self, image, scale_range=(1.0, 1.0), angle_range=(0, 0), translation_range=(0, 0), white_noise_percentage=0.0):
        """
        Apply augmentation to the image including scaling, rotation, translation, and adding white noise.

        Args:
            image (numpy.ndarray): The input image.
            scale_range (tuple): Range of scaling factors (min, max).
            angle_range (tuple): Range of rotation angles in degrees (min, max).
            translation_range (tuple): Range of translation values (min, max).
            white_noise_percentage (float): Percentage of pixels to be replaced with white noise.

        Returns:
            numpy.ndarray: The augmented image, flattened into a 1D array.
        """
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

    def train(self, X, y, epochs, learning_rate, batch_size, early_stopping_patience, graph=False, augment_settings=None):
        """
        Train the neural network using backpropagation and stochastic gradient descent.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target values.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            batch_size (int): Size of the batch for each iteration.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
            graph (bool): If True, plot the training loss and accuracy.
            augment_settings (dict, optional): Dictionary containing settings for data augmentation.

        Returns:
            None
        """
        loss_history = []
        accuracy_history = []

        best_accuracy = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                if augment_settings:
                    scale_range = augment_settings.get('scale_range', (1.0, 1.0))
                    angle_range = augment_settings.get('angle_range', (0, 0))
                    translation_range = augment_settings.get('translation_range', (0, 0))
                    white_noise_percentage = augment_settings.get('white_noise_percentage', 0.0)

                    X_batch = np.array([self.augment_image(image, scale_range, angle_range, translation_range, white_noise_percentage) for image in X_batch])

                hidden1 = self.sigmoid(np.dot(X_batch, self.weights_input_hidden1) + self.biases_hidden1)
                hidden2 = self.sigmoid(np.dot(hidden1, self.weights_hidden1_hidden2) + self.biases_hidden2)
                output = self.sigmoid(np.dot(hidden2, self.weights_hidden2_output) + self.biases_output)

                error = y_batch - output
                d_output = error * self.sigmoid_derivative(output)
                d_hidden2 = np.dot(d_output, self.weights_hidden2_output.T) * self.sigmoid_derivative(hidden2)
                d_hidden1 = np.dot(d_hidden2, self.weights_hidden1_hidden2.T) * self.sigmoid_derivative(hidden1)

                self.weights_hidden2_output += np.dot(hidden2.T, d_output) * learning_rate
                self.biases_output += np.sum(d_output, axis=0) * learning_rate
                self.weights_hidden1_hidden2 += np.dot(hidden1.T, d_hidden2) * learning_rate
                self.biases_hidden2 += np.sum(d_hidden2, axis=0) * learning_rate
                self.weights_input_hidden1 += np.dot(X_batch.T, d_hidden1) * learning_rate
                self.biases_hidden1 += np.sum(d_hidden1, axis=0) * learning_rate

            hidden1 = self.sigmoid(np.dot(X, self.weights_input_hidden1) + self.biases_hidden1)
            hidden2 = self.sigmoid(np.dot(hidden1, self.weights_hidden1_hidden2) + self.biases_hidden2)
            output = self.sigmoid(np.dot(hidden2, self.weights_hidden2_output) + self.biases_output)
            error = y - output
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            loss = self.cross_entropy(output, y)

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
            plt.savefig(f'loss_and_accuracy_{learning_rate}.png')
       
    def save_model(self, file_path):
        """
        Saves the model weights and biases to a file.

        Args:
            file_path (str): The path where the model should be saved.

        Returns:
            None
        """
        np.savez(file_path,
             weights_input_hidden1=self.weights_input_hidden1,
             biases_hidden1=self.biases_hidden1,
             weights_hidden1_hidden2=self.weights_hidden1_hidden2,
             biases_hidden2=self.biases_hidden2,
             weights_hidden2_output=self.weights_hidden2_output,
             biases_output=self.biases_output)

    def load_model(self, file_path):
        """
        Loads the model weights and biases from a file.

        Args:
            file_path (str): The path to the model file.

        Returns:
            None
        """
        data = np.load(file_path)
        self.weights_input_hidden1 = data['weights_input_hidden1']
        self.biases_hidden1 = data['biases_hidden1']
        self.weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
        self.biases_hidden2 = data['biases_hidden2']
        self.weights_hidden2_output = data['weights_hidden2_output']
        self.biases_output = data['biases_output']

    def load_data(self, directory):
        """
        Loads and preprocesses images and labels from a given directory.

        Args:
            directory (str): Path to the directory containing image data.

        Returns:
            tuple: A tuple containing arrays of images and labels.
        """
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

    def one_hot_encode(self, labels, num_classes):
        """
        Converts a vector of labels to one-hot encoded format.

        Args:
            labels (numpy.ndarray): Array of integer labels.
            num_classes (int): Number of classes for one-hot encoding.

        Returns:
            numpy.ndarray: One-hot encoded label array.
        """
        labels = np.asarray(labels, dtype=np.int_)

        # Initialize matrix of zeros
        one_hot = np.zeros((labels.size, num_classes))
        
        # np.arange(labels.size) creates an array with indices from 0 to the length of labels - 1
        # labels.reshape(-1) ensures that labels is a proper 1D array
        # This will then index the one_hot array and set the appropriate element to 1
        one_hot[np.arange(labels.size), labels.reshape(-1)] = 1
        return one_hot
    
