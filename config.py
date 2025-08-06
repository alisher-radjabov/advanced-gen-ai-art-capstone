import os
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    torch_dtype: torch.dtype = torch.float32
    enable_attention_slicing: bool = True
    
    # Paths
    input_dir: str = "input"
    output_dir: str = "output"
    resources_dir: str = "resources"
    original_image_name: str = "main.jpg"
    
    # Output filenames
    main_cover_output: str = "main_cover.jpg"
    generated_cover_output: str = "generated_album_cover.jpg"
    pipeline_screenshot_output: str = "pipeline_screenshot.jpg"
    report_output: str = "report.md"

class DeviceManager:
    """Manages device selection and optimization."""
    
    @staticmethod
    def get_optimal_device() -> torch.device:
        """Determine the best available device for generation."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def get_device_info(device: torch.device) -> dict:
        """Get information about the selected device."""
        info = {"type": device.type}
        
        if device.type == "cuda":
            info["name"] = torch.cuda.get_device_name(device)
            info["memory"] = f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB"
        elif device.type == "mps":
            info["name"] = "Apple Silicon"
            info["memory"] = "Shared system memory"
        else:
            info["name"] = "CPU"
            info["memory"] = "System RAM"
        
        return info