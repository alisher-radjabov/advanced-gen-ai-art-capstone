from PIL import Image
import os
from typing import Tuple
import logging
from config import GenerationConfig

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image loading, processing, and saving operations."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    def load_original_image(self) -> Image.Image:
        """Load and validate the original cover image."""
        original_path = os.path.join(self.config.input_dir, self.config.original_image_name)
        
        if not os.path.exists(original_path):
            raise FileNotFoundError(
                f"Please place the original album cover at {original_path}"
            )
        
        try:
            image = Image.open(original_path)
            logger.info(f"Loaded original image: {image.size}")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from {original_path}: {str(e)}")
    
    def prepare_dimensions(self, image: Image.Image) -> Tuple[int, int]:
        """Ensure image dimensions are compatible with Stable Diffusion (multiples of 8)."""
        width, height = image.size
        
        # Round down to nearest multiple of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        if width != image.size[0] or height != image.size[1]:
            logger.info(f"Adjusted dimensions from {image.size} to ({width}, {height})")
        
        return width, height
    
    def save_original_copy(self, image: Image.Image) -> str:
        """Save a copy of the original image to the output directory."""
        output_path = os.path.join(self.config.output_dir, self.config.main_cover_output)
        image.save(output_path)
        logger.info(f"Saved original copy to {output_path}")
        return output_path
    
    def save_generated_image(self, image: Image.Image) -> str:
        """Save the generated image."""
        output_path = os.path.join(self.config.output_dir, self.config.generated_cover_output)
        image.save(output_path)
        logger.info(f"Saved generated image to {output_path}")
        return output_path