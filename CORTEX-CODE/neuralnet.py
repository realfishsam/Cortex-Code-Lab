import numpy as np
import cv2
import os

class NeuralNetwork:
    def __init__(self):
        self.weights_input_hidden1 = None
        self.biases_hidden1 = None
        self.weights_hidden1_hidden2 = None
        self.biases_hidden2 = None
        self.weights_hidden2_output = None
        self.biases_output = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-15))
        return loss

    def augment_image(self, image, scale_range=(1.0, 1.0), angle_range=(0, 0), translation_range=(0, 0), white_noise_percentage=0.0):
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

    def train(self, X, y, epochs, learning_rate, batch_size, early_stopping_patience, graph=False, augment=False):
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

                if augment:
                    X_batch = np.array([self.augment_image(image) for image in X_batch])

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
        np.savez(file_path,
             weights_input_hidden1=weights_input_hidden1,
             biases_hidden1=biases_hidden1,
             weights_hidden1_hidden2=weights_hidden1_hidden2,
             biases_hidden2=biases_hidden2,
             weights_hidden2_output=weights_hidden2_output,
             biases_output=biases_output)

    def load_model(self, file_path):
        data = np.load(file_path)
        self.weights_input_hidden1 = data['weights_input_hidden1']
        self.biases_hidden1 = data['biases_hidden1']
        self.weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
        self.biases_hidden2 = data['biases_hidden2']
        self.weights_hidden2_output = data['weights_hidden2_output']
        self.biases_output = data['biases_output']

    def load_data(self, directory):
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
        labels = np.asarray(labels, dtype=np.int_)

        # Initialize matrix of zeros
        one_hot = np.zeros((labels.size, num_classes))
        
        # np.arange(labels.size) creates an array with indices from 0 to the length of labels - 1
        # labels.reshape(-1) ensures that labels is a proper 1D array
        # This will then index the one_hot array and set the appropriate element to 1
        one_hot[np.arange(labels.size), labels.reshape(-1)] = 1
        return one_hot
    
