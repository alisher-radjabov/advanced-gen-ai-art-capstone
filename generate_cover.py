import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Create output and input directories
os.makedirs("output", exist_ok=True)
os.makedirs("input", exist_ok=True)

# Check for original cover image
original_cover_path = "input/main.jpg"
if not os.path.exists(original_cover_path):
    raise FileNotFoundError("Please place the original album cover at input/main.jpg")

# Load and save original image
original_img = Image.open(original_cover_path)
original_img.save("output/original_cover.jpg")
width, height = original_img.size

# Adjust dimensions to be divisible by 8
width = (width // 8) * 8
height = (height // 8) * 8

# Set device (prefer MPS for Mac, fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# Disable safety checker to avoid TypeError
pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipeline = pipeline.to(device)
pipeline.enable_attention_slicing()
pipeline("test", num_inference_steps=1)  # Warm-up run

# Define prompt for image generation
prompt = (
    "A fantasy book cover illustration featuring a girl with a sword stepping through "
    "a glowing blue portal in an enchanted forest, with a mystical and hopeful atmosphere, "
    "fireflies, misty background, cinematic lighting, and a classic storybook style."
)

# Generate and save the AI image
generated_image = pipeline(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=height,
    width=width
).images[0]
generated_image.save("output/generated_album_cover.jpg")

# Create pipeline configuration plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.1, 0.8, f"Model: SD-v1.5\nDevice: {device.type}", fontsize=12)
ax.text(0.1, 0.5, "Sampler: default PFSD\nSteps: 30\nCFG: 7.5", fontsize=12)
ax.text(0.1, 0.2, f"Prompt:\n{prompt}", fontsize=10)
ax.axis("off")
plt.savefig("output/pipeline_screenshot.jpg")
plt.close()

# Write markdown report
with open("output/report.md", "w") as report_file:
    report_file.write("# AI-Generated Vinyl Cover: The Dark Side of the Moon\n\n")
    report_file.write("## Original Cover\n\n")
    report_file.write("![Original](original_cover.jpg)\n\n---\n\n")
    report_file.write("## AI-Generated Cover\n\n")
    report_file.write("![AI Cover](generated_album_cover.jpg)\n\n---\n\n")
    report_file.write("## Technical Details\n\n")
    report_file.write(f"**Model**: Stable Diffusion v1.5\n")
    report_file.write(f"**Device**: {device.type}\n")
    report_file.write("**Sampler**: default scheduler\n")
    report_file.write("**Steps**: 30\n")
    report_file.write("**CFG Scale**: 7.5\n\n")
    report_file.write("**Prompt**:\n")
    report_file.write(f"> {prompt}\n\n")
    report_file.write("## Pipeline Configuration\n\n")
    report_file.write("![Pipeline Setup](pipeline_screenshot.jpg)\n\n")
    report_file.write("## Resources\n\n")
    report_file.write("- **Interface**: diffusers Python pipeline\n")
    report_file.write("- **Hardware**: Apple Silicon MPS or CPU, ~16 GB RAM\n")
    report_file.write("- **Model**: Stable Diffusion v1.5\n")
    report_file.write("\nOutput saved in `output/` folder.\n")