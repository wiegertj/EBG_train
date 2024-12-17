from PIL import Image

# Open the image
path = "/Users/juliuswiegert/Downloads/Figure1_High_Res_(1).png"
img = Image.open(path)

# Get the original dimensions of the image
# Define the current DPI (assuming 300 DPI originally, you can replace with actual DPI if known)
current_dpi = 300

# Define the scaling factor (e.g., 0.8 to reduce the physical size by 20%)
scaling_factor = 0.8

# Calculate the new DPI to make the physical size smaller
new_dpi = int(current_dpi / scaling_factor)

# Save the image with the new DPI (pixels remain the same)
output_image_path = "/Users/juliuswiegert/Downloads/Figure1_High_Res__.png"
img.save(output_image_path, dpi=(600, 600))

print(f"Image saved with new DPI of {new_dpi} at {output_image_path}")