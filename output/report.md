# 🎵 AI Redesigned Vinyl Album Cover: The Dark Side of the Moon

## 💼 Original Cover

![Original](original_cover.jpg)

---

## 🎨 AI​-​Generated Variation

![AI Cover](generated_album_cover.jpg)

---

## ⚙️ Workflow & Technical Details

**Model**: Stable Diffusion v1.5 (Hugging Face) on self​-hosted setup
**Device**: mps
**LoRA**: Retro Album Art aesthetic (optional, not used here)
**Sampler/Backend**: default scheduler
**Steps**: 30
**CFG Scale**: 7.5

**Prompt**:
> An illustrated fantasy book cover showing a girl holding a sword as she steps through a radiant blue doorway set in the heart of an enchanted forest. The scene carries a sense of mystery and hope, with glowing fireflies, a misty backdrop, dramatic cinematic lighting, rich details, and a classic storybook art style.

## 📸 Pipeline Configuration Screenshot

![Pipeline Setup](pipeline_screenshot.jpg)

## 🧰 Resources Used

- **Interface**: `diffusers` Python pipeline (self​-hosted, no external API)
- **Hardware**: Apple Silicon MPS (or CPU fallback), ~16 GB RAM
- **Model**: Stable Diffusion v1.5
- **LoRA**: Retro aesthetic LoRA file (not loaded in this run)

👉 Folder `output/` contains the generated images and `report.md`.
