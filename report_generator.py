import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict
import logging
from config import GenerationConfig

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates reports and documentation for the generation process."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    def create_pipeline_screenshot(self, device_info: Dict, prompt: str) -> str:
        """Create a visual summary of the pipeline configuration."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        # Configuration text
        config_text = f"""Model: {self.config.model_name.split('/')[-1]}
Device: {device_info['type']} ({device_info['name']})
Memory: {device_info.get('memory', 'N/A')}

Sampler: Default PNDM
Steps: {self.config.num_inference_steps}
CFG Scale: {self.config.guidance_scale}
Torch dtype: {str(self.config.torch_dtype).split('.')[-1]}"""
        
        ax.text(0.05, 0.95, "🎨 Generation Configuration", 
                fontsize=16, fontweight='bold', transform=ax.transAxes, va='top')
        ax.text(0.05, 0.85, config_text, fontsize=11, 
                transform=ax.transAxes, va='top', family='monospace')
        
        # Prompt text (wrapped)
        prompt_lines = self._wrap_text(f"Prompt: {prompt}", 80)
        prompt_text = '\n'.join(prompt_lines)
        ax.text(0.05, 0.45, prompt_text, fontsize=10, 
                transform=ax.transAxes, va='top', style='italic')
        
        ax.axis('off')
        plt.tight_layout()
        
        output_path = os.path.join(self.config.output_dir, self.config.pipeline_screenshot_output)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created pipeline screenshot: {output_path}")
        return output_path
    
    def generate_markdown_report(self, device_info: Dict, prompt: str, 
                               generation_time: float) -> str:
        """Generate a comprehensive markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""# 🎵 AI-Generated Vinyl Album Cover

*Generated on {timestamp}*

## 💼 Original Cover

![Original Cover]({self.config.main_cover_output})

---

## 🎨 AI-Generated Variation

![AI Generated Cover]({self.config.generated_cover_output})

---

## ⚙️ Technical Details

| Parameter | Value |
|-----------|-------|
| **Model** | {self.config.model_name} |
| **Device** | {device_info['type']} ({device_info['name']}) |
| **Memory** | {device_info.get('memory', 'N/A')} |
| **Inference Steps** | {self.config.num_inference_steps} |
| **Guidance Scale** | {self.config.guidance_scale} |
| **Generation Time** | {generation_time:.2f} seconds |
| **Torch dtype** | {str(self.config.torch_dtype).split('.')[-1]} |

## 📝 Prompt Used

```
{prompt}
```

## 📸 Pipeline Configuration

![Pipeline Configuration]({self.config.pipeline_screenshot_output})

## 🧰 Resources & Tools

- **Framework**: Hugging Face Diffusers
- **Interface**: Custom Python pipeline (self-hosted)
- **Hardware**: {device_info['name']} with {device_info.get('memory', 'system memory')}
- **Model Source**: Hugging Face Model Hub

## 📁 Output Files

- `{self.config.main_cover_output}` - Original cover copy
- `{self.config.generated_cover_output}` - AI-generated variation
- `{self.config.pipeline_screenshot_output}` - Configuration summary
- `{self.config.report_output}` - This report

---

*Generated using Stable Diffusion pipeline*
"""
        
        output_path = os.path.join(self.config.output_dir, self.config.report_output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Generated report: {output_path}")
        return output_path
    
    @staticmethod
    def _wrap_text(text: str, width: int) -> list:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
