import numpy as np
import cv2

def sigmoid(x):
    # Clip the input values to prevent numerical overflow
    x_clipped = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x_clipped))

def load_model(file_path):
    # Load the neural network parameters from a .npz file
    data = np.load(file_path)
    weights_input_hidden1 = data['weights_input_hidden1']
    biases_hidden1 = data['biases_hidden1']
    weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
    biases_hidden2 = data['biases_hidden2']
    weights_hidden2_output = data['weights_hidden2_output']
    biases_output = data['biases_output']
    return weights_input_hidden1, biases_hidden1, weights_hidden1_hidden2, biases_hidden2, weights_hidden2_output, biases_output

def predict(image_path, model_path):
    # Load the image and process it
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Make sure the image is 28x28
    image = image.flatten() / 255.0  # Flatten and normalize

    # Load the model
    w_i_h1, b_h1, w_h1_h2, b_h2, w_h2_o, b_o = load_model(model_path)

    # Forward pass to get the prediction
    hidden1 = sigmoid(np.dot(image, w_i_h1) + b_h1)
    hidden2 = sigmoid(np.dot(hidden1, w_h1_h2) + b_h2)
    output = sigmoid(np.dot(hidden2, w_h2_o) + b_o)

    # The predicted class is the index with the highest probability
    prediction = np.argmax(output)
    return prediction

# Replace 'path_to_your_image.png' with the path to the image you want to predict
# Replace 'path_to_your_model.npz' with the path to your saved model
image_path = 'testSet/testSet/img_6.jpg'
model_path = 'my_neural_network.npz'
predicted_digit = predict(image_path, model_path)
print(f"The predicted digit is: {predicted_digit}")
