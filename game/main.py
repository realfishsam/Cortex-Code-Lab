import pygame
import random

def main():
    pygame.init()
    width, height = 1000, 800  # Adjusted window size for better layout
    canvas_resolution = (224, 224)  # High-res canvas resolution
    preview_resolution = (28, 28)  # Low-res preview resolution
    scale_factor = 10  # Factor by which we scale up the boxes to make them larger
    outline_color = (255, 255, 255)  # Color for the outline of the boxes
    outline_thickness = 2  # Thickness of the outline

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drawing Program")

    # Surfaces for high-res drawing and low-res preview
    canvas = pygame.Surface(canvas_resolution)
    canvas.fill((0, 0, 0))  # Start with a black canvas
    preview = pygame.Surface(preview_resolution)

    # Define button attributes
    button_color = (200, 200, 200)
    button_rect = pygame.Rect(width // 2 - 100, height - 100, 200, 50)  # Position the button at the bottom center
    button_text = 'Clear Canvas'

    drawing = False
    erasing = False
    brush_size = 5
    size_percentage = 100  # Arbitrary percentage for brush size
    last_pos = None  # Last position of the mouse

    # Define font for text
    font = pygame.font.SysFont(None, 36)

    def predict():
        return random.randint(0, 9)  # Returns a random integer between 0 and 9
    
    # Global variable to store the prediction
    prediction_number = None

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
            prediction_rect = prediction_text.get_rect(center=(3 * width // 4, preview_rect.bottom + 40))
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

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    clear_canvas()
                    prediction_number = predict()  # Call predict when the canvas is cleared
                else:
                    if event.button == 1:  # Left Click to Draw
                        drawing = True
                        prediction_number = predict()  # Call predict when drawing starts
                    elif event.button == 3:  # Right Click to Erase
                        erasing = True
                        prediction_number = predict()  # Call predict when erasing starts

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Stop Drawing
                    drawing = False
                elif event.button == 3:  # Stop Erasing
                    erasing = False
                last_pos = None  # Clear the last_pos on mouse up

            # Mouse wheel to change brush size
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:  # Scroll up to increase size
                    brush_size = min(brush_size + 1, 20)  # Cap the brush size to 20
                elif event.y < 0:  # Scroll down to decrease size
                    brush_size = max(brush_size - 1, 1)  # Ensure the brush size doesn't go below 1

            if event.type == pygame.MOUSEMOTION:
                mouse_position = event.pos
                if drawing or erasing:
                    color = (255, 255, 255) if drawing else (0, 0, 0)
                    if last_pos:
                        # Adjust coordinates for drawing on the canvas
                        canvas_pos = (last_pos[0] - canvas_rect.left, last_pos[1] - canvas_rect.top)
                        new_pos = (mouse_position[0] - canvas_rect.left, mouse_position[1] - canvas_rect.top)
                        pygame.draw.line(canvas, color, canvas_pos, new_pos, brush_size)
                    last_pos = mouse_position
                    prediction_number = predict()

        size_percentage = int((brush_size / 20.0) * 100)

        # Clear the screen and blit the canvas and the preview to the screen surface
        screen.fill((0, 0, 0))

        # Display the high-res canvas
        canvas_rect = canvas.get_rect(center=(width // 4, height // 2))
        screen.blit(canvas, canvas_rect.topleft)
        pygame.draw.rect(screen, outline_color, canvas_rect.inflate(outline_thickness * 2, outline_thickness * 2), outline_thickness)
        input_text = font.render('Input', True, outline_color)
        screen.blit(input_text, (canvas_rect.left, canvas_rect.top - 40))

        update_preview()
        display_brush_size()
        display_button()

        pygame.display.flip()

if __name__ == "__main__":
    main()