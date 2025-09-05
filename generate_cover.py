
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline.to("cpu")

prompt = "a CD album cover for a rock band, dark colors, skull motif, digital art"
image = pipeline(prompt).images[0]

image.save("cd_cover.png")


