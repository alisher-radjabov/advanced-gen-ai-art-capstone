# AI Media Cover Redesign – Vinyl Album Project

This project generates an AI-redesigned version of an iconic media cover using a self-hosted image generation model (Stable Diffusion v1.5).

# Media Type
**Vinyl Album** – themed around classic psychedelic/fantasy aesthetics

# Tools
- **Model**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Library**: 🤗 HuggingFace `diffusers`
- **Hardware**: macOS with Apple M1/M2 (MPS) or CPU fallback
- **Prompt**: Inspired by the original cover — surrealism, glowing door, forest themes

# Output
- `main_cover.jpg`: the original image placed in `input/`
- `album_cover.jpg`: AI-generated alternative cover
- `pipeline_screenshot.jpg`: visual summary of pipeline parameters
- `report.md`: full markdown report documenting the generation process

# Run Instructions

1. Place your original image here:
```
input/main.jpg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python generate_cover.py
```

4. Output files will be saved in the `output/` directory.

## 🧰 Project Structure

```
├── generate_cover.py        # Main script
├── input/
│   └── main.jpg               # Input image (you must add manually)
├── output/
│   ├── main_cover.jpg
│   ├── generated_album_cover.jpg
│   ├── pipeline_screenshot.jpg
│   └── report.md
```

## ✅ Requirements Met

- [x] Uses self-hosted image generation (no online APIs)
- [x] Original and AI-generated works
- [x] Full workflow documentation
- [x] Local runtime, reproducible output

---

© 2025 – Generated with 🧠 & 🎨 on self-hosted Stable Diffusion.
