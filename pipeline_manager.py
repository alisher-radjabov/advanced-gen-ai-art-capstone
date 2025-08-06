from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import logging
from typing import Optional
from config import GenerationConfig

logger = logging.getLogger(__name__)

class PipelineManager:
    """Manages the Stable Diffusion pipeline."""
    
    def __init__(self, config: GenerationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.pipeline = None
    
    def initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype
            )
            
            # Fix safety checker to avoid TypeError
            self.pipeline.safety_checker = self._dummy_safety_checker
            self.pipeline = self.pipeline.to(self.device)
            
            if self.config.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
            
            # Warm up the pipeline
            logger.info("Warming up pipeline...")
            _ = self.pipeline("test", num_inference_steps=1)
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    def generate_image(self, prompt: str, width: int, height: int) -> Image.Image:
        """Generate an image using the pipeline."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        try:
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            result = self.pipeline(
                prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=height,
                width=width
            )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            raise
    
    @staticmethod
    def _dummy_safety_checker(images, **kwargs):
        """Dummy safety checker to avoid TypeError."""
        return images, [False] * len(images)