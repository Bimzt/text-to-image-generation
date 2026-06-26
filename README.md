# Text-to-Image Generation with Stable Diffusion & SAM

> A Generative AI pipeline combining **Stable Diffusion** for text-to-image synthesis and **Segment Anything Model (SAM)** for intelligent image segmentation packaged into an interactive **Streamlit** web application.

---

## Overview

This project explores the frontier of **multimodal Generative AI** by building an end-to-end pipeline that goes beyond simple text-to-image generation. By integrating Stable Diffusion with Meta's Segment Anything Model (SAM), the system enables not just image synthesis from natural language prompts, but also precise object-level manipulation through segmentation-guided inpainting.

Two deliverables are produced:
- **`Pipeline_GenAI_Project.ipynb`** core diffusion pipeline: prompt engineering, image generation, and SAM-based segmentation
- **`Streamlit_GenAI_Project.ipynb`** interactive web app built with Streamlit and a drawable canvas for real-time image editing

---

## Key Capabilities

- **Text-to-Image Generation** - synthesize photorealistic images from natural language prompts using Stable Diffusion
- **Semantic Segmentation** - isolate any object within a generated image using SAM (zero-shot, prompt-free masking)
- **Inpainting / Image Editing** - replace or modify masked regions guided by new text prompts
- **Interactive Web App** - drawable canvas interface for user-defined segmentation masks via Streamlit

---

## Repository Structure

```
├── Pipeline_GenAI_Project.ipynb       # Core Gen AI pipeline (generation + SAM segmentation)
├── Streamlit_GenAI_Project.ipynb      # Streamlit web app with interactive canvas
├── requirements.txt                   # All dependencies with pinned versions
└── README.md
```

---

## Architecture

```
Text Prompt
    │
    ▼
┌─────────────────────────────────┐
│     Stable Diffusion v2         │  ← diffusers + xformers (memory-efficient attention)
│  (Text → Latent → Image)        │
└─────────────────────────────────┘
    │
    ▼
Generated Image
    │
    ├──► [Pipeline Notebook]
    │         │
    │         ▼
    │    SAM (Segment Anything)       ← Zero-shot object segmentation
    │         │
    │         ▼
    │    Segmentation Mask
    │         │
    │         ▼
    │    Inpainting Pipeline          ← Stable Diffusion Inpaint
    │         │
    │         ▼
    │    Edited Image Output
    │
    └──► [Streamlit App]
              │
              ▼
         Drawable Canvas             ← streamlit-drawable-canvas
              │
              ▼
         User-defined Mask
              │
              ▼
         Inpainting Result
```

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Diffusion Model | `diffusers` | 0.27.2 |
| Language Model Backend | `transformers` | 4.40.0 |
| Deep Learning | `torch` + `torchvision` | 2.3.0 / 0.18.0 |
| Memory-Efficient Attention | `xformers` | 0.0.26.post1 |
| Segmentation | `segment-anything` (SAM) | 1.0 |
| Web App | `streamlit` | 1.35.0 |
| Interactive Canvas | `streamlit-drawable-canvas` | 0.8.0 |
| Tunneling (Colab) | `pyngrok` | 7.2.0 |
| Image Processing | `Pillow`, `numpy`, `scipy` | — |
| Watermarking | `invisible-watermark` | 0.2.0 |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Bimzt/text-to-image-generation.git
cd text-to-image-generation
```

### 2. Install dependencies

> Requires **Python 3.9+** and a **CUDA-capable GPU** (minimum 8GB VRAM recommended). CPU inference is possible but very slow.

```bash
pip install -r requirements.txt
```

### 3a. Run the pipeline notebook
```bash
jupyter notebook Pipeline_GenAI_Project.ipynb
```

### 3b. Run the Streamlit app (via Google Colab + ngrok)

Open `Streamlit_GenAI_Project.ipynb` in Google Colab, set your ngrok auth token, and run all cells. The public URL will be printed in the output.

```python
# Inside the notebook:
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
```

> Get a free ngrok token at [https://dashboard.ngrok.com](https://dashboard.ngrok.com)

---

## How It Works

### 1. Text-to-Image (Stable Diffusion)
The diffusion model starts from random Gaussian noise in latent space and iteratively denoises it, conditioned on a CLIP text embedding of the input prompt. `xformers` is enabled for memory-efficient attention, allowing generation at higher resolutions.

### 2. Segmentation (SAM — Segment Anything Model)
SAM from Meta AI performs zero-shot segmentation no training or fine-tuning needed. Given the generated image, SAM produces pixel-accurate masks for any object, either via automatic mask generation or point/box prompts.

### 3. Inpainting
The segmentation mask is fed back into Stable Diffusion's inpainting pipeline. A new text prompt guides what fills the masked region, enabling object replacement, background swapping, or style transfer on selected regions.

### 4. Streamlit Interactive App
The web app provides a drawable canvas where users can manually brush masks over any region of the generated image, then trigger inpainting with a new text prompt all without writing code.

---

## Real-World Impact

Text-to-image generation with segmentation-guided editing represents one of the most commercially relevant areas in Generative AI today:

- **Creative & Design Industry** - Enables designers and illustrators to rapidly prototype visual concepts from text descriptions, reducing ideation time from days to minutes
- **E-Commerce & Advertising** - Automates product image generation and background replacement for catalog photography at scale, eliminating costly photo shoots
- **Game & Film Production** - Accelerates asset generation for concept art, texture creation, and scene prototyping; studios can explore hundreds of visual directions before committing to production
- **Medical Imaging (Future Direction)** - SAM's zero-shot segmentation capability translates directly to organ/lesion isolation in medical scans, and diffusion models show promise for synthetic training data generation
- **Accessibility** - Empowers non-designers to create professional-grade visuals through natural language, democratizing visual content creation for small businesses and individuals
- **Education & Research** - Demonstrates end-to-end understanding of latent diffusion models, attention mechanisms, and foundation model composability core competencies for modern AI/ML roles