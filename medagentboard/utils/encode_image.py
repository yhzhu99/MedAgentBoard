import base64
import os

def encode_image(image_path: str) -> str:
    """
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the image file
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading image file: {e}")