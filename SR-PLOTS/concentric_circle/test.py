import os
import math
from PIL import Image

def create_image_grid(input_folder, output_folder, grid_rows=6, grid_cols=3, max_images=None):
    """
    Create grid images from images in an input folder.
    
    Args:
    - input_folder: Path to folder containing source images
    - output_folder: Path to folder where grid images will be saved
    - grid_rows: Number of rows in the grid (default 3)
    - grid_cols: Number of columns in the grid (default 6)
    - max_images: Maximum number of images to process (optional)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Limit images if max_images is specified
    if max_images:
        image_files = image_files[:max_images]
    
    # Calculate number of grid images needed
    grid_size = grid_rows * grid_cols
    num_grid_images = math.ceil(len(image_files) / grid_size)
    
    # Open and resize images
    images = []
    for filename in image_files:
        try:
            img = Image.open(os.path.join(input_folder, filename))
            images.append(img)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create grid images
    for grid_num in range(num_grid_images):
        # Create a new blank image for the grid
        # Assume first image's mode and size for resizing
        first_img = images[0]
        target_width = first_img.width
        target_height = first_img.height
        
        # Create a new white image for the grid
        grid_width = target_width * grid_cols
        grid_height = target_height * grid_rows
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Place images in the grid
        for i in range(grid_size):
            img_index = grid_num * grid_size + i
            
            # Stop if we've run out of images
            if img_index >= len(images):
                break
            
            # Calculate grid position
            row = i // grid_cols
            col = i % grid_cols
            
            # Resize image to target size
            current_img = images[img_index].resize((target_width, target_height))
            
            # Paste image into grid
            x = col * target_width
            y = row * target_height
            grid_image.paste(current_img, (x, y))
        
        # Save the grid image
        output_filename = f'grid_{grid_num + 1}.jpg'
        grid_image.save(os.path.join(output_folder, output_filename))
        print(f"Created {output_filename}")

# Example usage
if __name__ == "__main__":
    input_folder = '../single_variable_classification/NEUROCHAOS-RESULTS'  # folder containing source images
    output_folder = '../single_variable_classification/grid_result_2'  # folder to save grid images
    
    create_image_grid(input_folder, output_folder)