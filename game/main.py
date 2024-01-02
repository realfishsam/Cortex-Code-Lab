import pygame
import random
import numpy as np
import cv2
import pprint

# Neural network helper functions
def sigmoid(x):
    # Clip the input values to prevent numerical overflow
    x_clipped = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x_clipped))

def load_model(file_path):
    # Load the neural network parameters from a .npz file
    data = np.load(file_path, allow_pickle=True)
    weights_input_hidden1 = data['weights_input_hidden1']
    biases_hidden1 = data['biases_hidden1']
    weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
    biases_hidden2 = data['biases_hidden2']
    weights_hidden2_output = data['weights_hidden2_output']
    biases_output = data['biases_output']
    return weights_input_hidden1, biases_hidden1, weights_hidden1_hidden2, biases_hidden2, weights_hidden2_output, biases_output

def find_centroid(image):
    # Convert to a binary image, white areas (digit) are True
    binary_image = image > 0
    # Calculate the coordinates of the center of mass of the white areas
    coords = np.column_stack(np.where(binary_image))
    if coords.size == 0:
        return None, None  # If the image is completely black, return None
    centroid = coords.mean(axis=0)
    return int(centroid[1]), int(centroid[0])  # return (cx, cy)

def center_digit(image, size):
    # Find the centroid of the digit
    cx, cy = find_centroid(image)
    if cx is None:
        return image  # If the image is completely black, return the original image

    # Calculate the shift needed to center the digit
    shift_x = size[1] // 2 - cx
    shift_y = size[0] // 2 - cy

    # Apply the shift to the image using affine transformation
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_image = cv2.warpAffine(image, M, (size[1], size[0]))

    return centered_image

def neural_predict(canvas_surface, model_path, verbose=False):
    # Convert the canvas surface to a string buffer and then to a numpy array
    canvas_string = pygame.image.tostring(canvas_surface, 'RGB')
    canvas_np = np.frombuffer(canvas_string, np.uint8).reshape((224, 224, 3))

    # Convert from RGB to grayscale
    canvas_gray = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2GRAY)

    # Center the digit on the canvas
    canvas_centered = center_digit(canvas_gray, canvas_gray.shape)

    # Resize to 28x28
    canvas_resized = cv2.resize(canvas_centered, (28, 28))

    # Normalize the pixel values
    canvas_normalized = canvas_resized.flatten() / 255.0

    # Load the model
    w_i_h1, b_h1, w_h1_h2, b_h2, w_h2_o, b_o = load_model(model_path)

    # Forward pass to get the prediction
    hidden1 = sigmoid(np.dot(canvas_normalized, w_i_h1) + b_h1)
    hidden2 = sigmoid(np.dot(hidden1, w_h1_h2) + b_h2)
    output = sigmoid(np.dot(hidden2, w_h2_o) + b_o)

    if verbose:
        # map each output to a number, eg. 0 goes to index 0, and pprint the output and the prediction
        output_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
        pprint.pprint({output_map[i]: output[i] for i in range(len(output))})
        print()

    # The predicted class is the index with the highest probability
    prediction = np.argmax(output)
    return prediction

def main():
    pygame.init()
    width, height = 1000, 800
    canvas_resolution = (224, 224)
    preview_resolution = (28, 28)
    scale_factor = 8  # Adjusted scale factor to match visual size
    outline_color = (255, 255, 255)
    outline_thickness = 2

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drawing Program")

    canvas = pygame.Surface(canvas_resolution)
    canvas.fill((0, 0, 0))
    preview = pygame.Surface(preview_resolution)

    button_color = (200, 200, 200)
    button_rect = pygame.Rect(width // 2 - 100, height - 100, 200, 50)
    button_text = 'Clear Canvas'

    drawing = False
    erasing = False
    brush_size = 25  # Max brush size indicating 100%
    size_percentage = 100
    last_pos = None

    font = pygame.font.SysFont(None, 36)
    
    # Replace this path with the actual path to your neural network model file
    model_path = '82.npz'
    prediction_number = None

    def predict():
        return random.randint(0, 9)  # Returns a random integer between 0 and 9

    # Function to display brush size on the screen
    def display_brush_size():
        text = font.render(f"Size: {size_percentage}%", True, outline_color)
        screen.blit(text, (10, 10))

    # Function to update the preview
    def update_preview():
        scaled_preview = pygame.transform.smoothscale(canvas, preview_resolution)
        preview.blit(scaled_preview, (0, 0))
        
        # Updated scale factor to match the visual size of the canvas
        scale_factor = 8  # New scale factor to match the canvas size
        
        scaled_up_preview = pygame.transform.scale(preview, (preview_resolution[0] * scale_factor, preview_resolution[1] * scale_factor))
        preview_rect = scaled_up_preview.get_rect(center=(3 * width // 4, height // 2))
        screen.blit(scaled_up_preview, preview_rect.topleft)
        pygame.draw.rect(screen, outline_color, preview_rect.inflate(outline_thickness * 2, outline_thickness * 2), outline_thickness)
        output_text = font.render('Output', True, outline_color)
        screen.blit(output_text, (preview_rect.left, preview_rect.top - 40))

        if prediction_number is not None:
            prediction_text = font.render(f"Prediction: {prediction_number}", True, outline_color)
            prediction_rect = prediction_text.get_rect(center=(3 * width // 4, height // 2 + canvas_resolution[1] // 2 + 20))
            screen.blit(prediction_text, prediction_rect.topleft)

    # Function to display the button
    def display_button():
        pygame.draw.rect(screen, button_color, button_rect)
        text = font.render(button_text, True, (0, 0, 0))
        text_rect = text.get_rect(center=button_rect.center)
        screen.blit(text, text_rect)

    # Function to clear the canvas
    def clear_canvas():
        canvas.fill((0, 0, 0))
        global prediction_number
        prediction_number = None

    def is_canvas_blank(canvas_surface):
        canvas_array = pygame.surfarray.array3d(canvas_surface)
        return np.all(canvas_array == 0)

    while True:                  
        if is_canvas_blank(canvas):
            prediction_number = None
        else:
            prediction_number = neural_predict(canvas, model_path)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    clear_canvas()
                else:
                    if event.button == 1:  # Left Click to Draw
                        drawing = True
                    elif event.button == 3:  # Right Click to Erase
                        erasing = True
                    last_pos = event.pos  # Update last_pos for new drawing/erasing

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button in [1, 3]:  # Left Click or Right Click
                    drawing = False
                    erasing = False
                    last_pos = None

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:  # Scroll up to increase size
                    brush_size = min(brush_size + 1, 20)
                elif event.y < 0:  # Scroll down to decrease size
                    brush_size = max(brush_size - 1, 1)
                size_percentage = int((brush_size / 20.0) * 100)  # Update size percentage
                if size_percentage > 100:
                    size_percentage = 100


            if event.type == pygame.MOUSEMOTION:
                if drawing or erasing:
                    color = (255, 255, 255) if drawing else (0, 0, 0)
                    if last_pos:
                        # Calculate the canvas-relative coordinates
                        canvas_pos = (last_pos[0] - canvas_rect.left, last_pos[1] - canvas_rect.top)
                        new_pos = (event.pos[0] - canvas_rect.left, event.pos[1] - canvas_rect.top)
                        # Draw a line on the canvas Surface
                        pygame.draw.line(canvas, color, canvas_pos, new_pos, brush_size)
                    # Update the last_pos with the current mouse position
                    last_pos = event.pos

        # Clear the screen and blit the canvas and the preview to the screen surface
        screen.fill((0, 0, 0))
        canvas_rect = canvas.get_rect(center=(width // 4, height // 2))
        screen.blit(canvas, canvas_rect.topleft)
        pygame.draw.rect(screen, outline_color, canvas_rect.inflate(outline_thickness * 2, outline_thickness * 2), outline_thickness)
        input_text = font.render('Input', True, outline_color)
        screen.blit(input_text, (canvas_rect.left, canvas_rect.top - 40))

        # Update the preview and prediction display
        update_preview()
        if prediction_number is not None:
            prediction_text = font.render(f"Prediction: {prediction_number}", True, outline_color)
            prediction_rect = prediction_text.get_rect(center=(3 * width // 4, height // 2 + canvas_resolution[1] // 2 + 20))
            screen.blit(prediction_text, prediction_rect.topleft)

        # Display brush size and the button
        display_brush_size()
        display_button()

        pygame.display.flip()

if __name__ == "__main__":
    main()
    