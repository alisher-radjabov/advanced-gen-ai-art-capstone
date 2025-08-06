import os
import logging
import time
from pathlib import Path

from config import GenerationConfig, DeviceManager
from image_processor import ImageProcessor
from prompt_manager import PromptManager, Style
from pipeline_manager import PipelineManager
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VinylCoverGenerator:
    """Main class for generating AI vinyl covers."""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.device = DeviceManager.get_optimal_device()
        self.device_info = DeviceManager.get_device_info(self.device)
        
        # Initialize components
        self.image_processor = ImageProcessor(self.config)
        self.pipeline_manager = PipelineManager(self.config, self.device)
        self.report_generator = ReportGenerator(self.config)
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.input_dir,
            self.config.output_dir,
            self.config.resources_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def generate_cover(self, 
                      prompt: str = None, 
                      style: Style = Style.FANTASY,
                      custom_elements: list = None) -> dict:
        """Generate a vinyl cover with the specified parameters."""
        start_time = time.time()
        
        try:
            # Load and prepare original image
            logger.info("Loading original image...")
            original_image = self.image_processor.load_original_image()
            width, height = self.image_processor.prepare_dimensions(original_image)
            
            # Save original copy
            original_copy_path = self.image_processor.save_original_copy(original_image)
            
            # Generate or use provided prompt
            if prompt is None:
                if custom_elements:
                    prompt = PromptManager.create_prompt(
                        "an album cover design", style, custom_elements
                    )
                else:
                    prompt = PromptManager.get_default_prompt()
            
            logger.info(f"Using prompt: {prompt}")
            
            # Initialize and use pipeline
            logger.info("Initializing AI pipeline...")
            self.pipeline_manager.initialize_pipeline()
            
            logger.info("Generating new cover...")
            generated_image = self.pipeline_manager.generate_image(prompt, width, height)
            
            # Save generated image
            generated_path = self.image_processor.save_generated_image(generated_image)
            
            generation_time = time.time() - start_time
            
            # Create documentation
            logger.info("Creating documentation...")
            screenshot_path = self.report_generator.create_pipeline_screenshot(
                self.device_info, prompt
            )
            report_path = self.report_generator.generate_markdown_report(
                self.device_info, prompt, generation_time
            )
            
            result = {
                'success': True,
                'original_path': original_copy_path,
                'generated_path': generated_path,
                'screenshot_path': screenshot_path,
                'report_path': report_path,
                'generation_time': generation_time,
                'prompt': prompt,
                'device_info': self.device_info
            }
            
            logger.info(f"Generation completed successfully in {generation_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time
            }

def main():
    """Main function to run the vinyl cover generator."""
    # You can customize these parameters
    config = GenerationConfig(
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    generator = VinylCoverGenerator(config)
    
    # Generate with default settings
    result = generator.generate_cover()
    
    if result['success']:
        print(f"\n✅ Generation completed successfully!")
        print(f"⏱️  Time taken: {result['generation_time']:.2f} seconds")
        print(f"🖥️  Device used: {result['device_info']['name']}")
        print(f"📁 Check the '{config.output_dir}' folder for results")
        print(f"📝 Report: {result['report_path']}")
    else:
        print(f"\n❌ Generation failed: {result['error']}")

if __name__ == "__main__":
    main()