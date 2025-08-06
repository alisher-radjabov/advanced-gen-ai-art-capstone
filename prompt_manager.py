from typing import List, Dict
from enum import Enum

class Style(Enum):
    FANTASY = "fantasy"
    RETRO = "retro"
    ABSTRACT = "abstract"
    MINIMALIST = "minimalist"
    CYBERPUNK = "cyberpunk"

class PromptManager:
    """Manages prompt generation and templates."""
    
    STYLE_TEMPLATES = {
        Style.FANTASY: {
            "elements": ["magical forest", "glowing elements", "mystical atmosphere"],
            "lighting": "cinematic lighting, soft ethereal glow",
            "quality": "highly detailed, fantasy art style"
        },
        Style.RETRO: {
            "elements": ["vintage aesthetic", "retro colors", "classic design"],
            "lighting": "warm nostalgic lighting",
            "quality": "vintage poster style, high quality"
        },
        Style.ABSTRACT: {
            "elements": ["geometric shapes", "flowing forms", "abstract composition"],
            "lighting": "dramatic lighting, color gradients",
            "quality": "modern abstract art, professional"
        },
        Style.MINIMALIST: {
            "elements": ["clean lines", "simple composition", "negative space"],
            "lighting": "soft even lighting",
            "quality": "minimalist design, elegant"
        },
        Style.CYBERPUNK: {
            "elements": ["neon lights", "futuristic city", "digital elements"],
            "lighting": "neon lighting, dramatic shadows",
            "quality": "cyberpunk aesthetic, high detail"
        }
    }
    
    @classmethod
    def create_prompt(cls, 
                     subject: str, 
                     style: Style = Style.FANTASY,
                     custom_elements: List[str] = None) -> str:
        """Create a detailed prompt based on subject and style."""
        template = cls.STYLE_TEMPLATES[style]
        
        elements = template["elements"].copy()
        if custom_elements:
            elements.extend(custom_elements)
        
        prompt_parts = [
            f"A {style.value} album cover illustration of {subject}",
            ", ".join(elements),
            template["lighting"],
            template["quality"]
        ]
        
        return ", ".join(prompt_parts)
    
    @classmethod
    def get_default_prompt(cls) -> str:
        """Get the default fantasy prompt."""
        return cls.create_prompt(
            subject="a girl with a sword stepping into a glowing blue door",
            style=Style.FANTASY,
            custom_elements=["fireflies", "misty background", "mysterious and hopeful atmosphere"]
        )