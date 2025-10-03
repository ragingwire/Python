import os
from rembg import remove
from PIL import Image

def remove_image_background(input_path, output_path):
    """
    Removes the background from an image using the rembg library.

    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path where the output image (with background removed)
                           will be saved.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at '{input_path}'")
            return

        # Open the input image
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                # Read the input image data
                input_data = i.read()
                
                # Remove the background
                # The 'remove' function returns bytes data of the processed image
                output_data = remove(input_data)
                
                # Write the output data to the specified file
                o.write(output_data)
        
        print(f"Background successfully removed. Output saved to '{output_path}'")

        # Optional: Display the original and processed images (requires Pillow)
        try:
            original_image = Image.open(input_path)
            processed_image = Image.open(output_path)

            print("\nDisplaying images (close windows to continue):")
            original_image.show(title="Original Image")
            processed_image.show(title="Processed Image (Background Removed)")
        except Exception as e:
            print(f"Could not display images: {e}. Ensure you have a display environment.")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---


# Define input and output paths
input_image_path = "F:\\downloads\\input.png"
output_image_path = 'F:\\downloads\\test.png'

# Call the function to remove the background
remove_image_background(input_image_path, output_image_path)

# Clean up the dummy input image (optional)
if os.path.exists(input_image_path):
    #os.remove(input_image_path)
    print("\nCleaned up dummy input_image.png.")
