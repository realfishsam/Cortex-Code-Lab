import pygame
import random
import numpy as np
import cv2
import pprint
import sys

# find the path to the current file
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{path}/CORTEX_CODE/')
from neuralnet import NeuralNetwork

def find_centroid(image):
    """
    Calculate the centroid of the white areas in a binary image.
    This function converts the given image to a binary image where the white areas 
    (digit) are marked as True. It then calculates the coordinates of the center of 
    mass of these white areas. If the image is completely black, it returns None.
    
    Parameters:
    image (numpy.ndarray): The image in which the centroid is to be found.

    Returns:
    tuple: The coordinates (x, y) of the centroid, or (None, None) if the image is black.
    """
    # Convert to a binary image, white areas (digit) are True
    binary_image = image > 0
    # Calculate the coordinates of the center of mass of the white areas
    coords = np.column_stack(np.where(binary_image))
    if coords.size == 0:
        return None, None  # If the image is completely black, return None
    centroid = coords.mean(axis=0)
    return int(centroid[1]), int(centroid[0])  # return (cx, cy)

def center_digit(image, size):
    """
    Center the digit in the given image.
    This function finds the centroid of the digit in the image and calculates the necessary
    shift to center the digit. It then applies this shift using an affine transformation.
    If the image is completely black, it returns the original image.
    
    Parameters:
    image (numpy.ndarray): The image containing the digit to be centered.
    size (tuple): The dimensions (height, width) to which the image should be centered.

    Returns:
    numpy.ndarray: The centered image.
    """
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
    brush_size = 13
    size_percentage = 50
    last_pos = None

    font = pygame.font.SysFont(None, 36)
    
    model_path = '82.npz'
    neural_net = NeuralNetwork()  # Create an instance of the NeuralNetwork
    neural_net.load_model(model_path)  # Load the model

    prediction_number = None

    # Update the preview and prediction display
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

    def display_brush_size():
        text = font.render(f"Size: {size_percentage}%", True, outline_color)
        screen.blit(text, (10, 10))

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

    def neural_predict(canvas_surface, verbose=False):
        canvas_string = pygame.image.tostring(canvas_surface, 'RGB')
        canvas_np = np.frombuffer(canvas_string, np.uint8).reshape((224, 224, 3))
        canvas_gray = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2GRAY)
        canvas_centered = center_digit(canvas_gray, canvas_gray.shape)
        canvas_resized = cv2.resize(canvas_centered, (28, 28))
        canvas_normalized = canvas_resized.flatten() / 255.0

        hidden1 = neural_net.sigmoid(np.dot(canvas_normalized, neural_net.weights_input_hidden1) + neural_net.biases_hidden1)
        hidden2 = neural_net.sigmoid(np.dot(hidden1, neural_net.weights_hidden1_hidden2) + neural_net.biases_hidden2)
        output = neural_net.sigmoid(np.dot(hidden2, neural_net.weights_hidden2_output) + neural_net.biases_output)

        if verbose:
            output_map = {i: str(output[i]) for i in range(10)}
            pprint.pprint(output_map)
            print()

        prediction = np.argmax(output)
        return prediction
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:  # Scroll up to increase size
                    brush_size = min(brush_size + 1, 20)
                elif event.y < 0:  # Scroll down to decrease size
                    brush_size = max(brush_size - 1, 1)
                size_percentage = int((brush_size / 20.0) * 100)  # Update size percentage
                if size_percentage > 100:
                    size_percentage = 100

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):  # Check if the button is clicked
                    clear_canvas()
                elif canvas.get_rect(center=(width // 4, height // 2)).collidepoint(event.pos):
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

            if event.type == pygame.MOUSEMOTION:
                if drawing or erasing:
                    color = (255, 255, 255) if drawing else (0, 0, 0)
                    if last_pos:
                        # Calculate the canvas-relative coordinates
                        canvas_pos = (last_pos[0] - canvas_rect.left, last_pos[1] - canvas_rect.top)
                        new_pos = (event.pos[0] - canvas_rect.left, event.pos[1] - canvas_rect.top)
                        # Draw a line on the canvas Surface
                        pygame.draw.line(canvas, color, canvas_pos, new_pos, brush_size)
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

        if not is_canvas_blank(canvas):
            prediction_number = neural_predict(canvas)

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
    