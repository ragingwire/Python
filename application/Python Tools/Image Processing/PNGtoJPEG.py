import os
import sys
from PIL import Image



def convert_png_to_jpeg(input_dir, output_dir, quality, converted_count, skipped_count ):

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return False # Return False to indicate failure

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting PNGs from '{input_dir}' to JPEGs in '{output_dir}'...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            png_path = os.path.join(input_dir, filename)
            png_creation_timestamp = os.path.getctime( png_path )
            # Create a new filename for the JPEG
            jpeg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpeg_path = os.path.join(output_dir, jpeg_filename)

            try:
                with Image.open(png_path) as img:
                    # Convert to RGB mode if the image has an alpha channel (RGBA)
                    # JPEGs do not support alpha channels, so converting to RGB
                    # prevents black backgrounds or transparency issues.
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    #img.show ()
                    img.save(jpeg_path, 'jpeg', quality=quality)
                    #os.utime ( jpeg_path, ( png_creation_timestamp, npng_creation_timestamp ))
                    print(f"Converted '{filename}' to '{jpeg_filename}'")
                    converted_count += 1
            except Exception as e:
                print(f"Error converting '{filename}': {e}")
                skipped_count += 1
        else:
            print(f"Skipping non-PNG file: '{filename}'")
            skipped_count += 1

    print(f"\nConversion run complete!")
    print(f"Total files processed in input directory: {converted_count + skipped_count}")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped/Errors: {skipped_count}")
    return True # Return True to indicate success


# --- Example Usage ---
if __name__ == "__main__":
    # Check if a directory path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <input_directory_path>")
        print("Example: python png_converter.py C:/Users/YourUser/Pictures/MyPNGs")
        sys.exit(1) # Exit with an error code

    # Get the input directory from the command line argument
    input_directory = sys.argv[1]
    if len ( sys.argv ) > 2:
        output_directory = sys.argv [2]
    else:
        output_directory = input_directory
    

    jpeg_quality = 95
    converted_count = 0
    skipped_count = 0
    # Call the conversion function
    success = convert_png_to_jpeg(input_directory, output_directory, jpeg_quality, converted_count, skipped_count )

    if success:
        if converted_count > 0:
            print(f"\nAll converted JPEG images are located in the '{output_directory}' directory.")
        else:
            print("\nNothing to do.")
            
    else:
        print("\nConversion failed due to an error. Please check the messages above.")
        
    input("\nPress any key to end program ")
   