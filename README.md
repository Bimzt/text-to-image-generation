# Text-to-Image Generation, Segmentation & Editing (Stable Diffusion v1.5 + SAM)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.27%2B-FFD21E)](https://huggingface.co/docs/diffusers)
[![SAM](https://img.shields.io/badge/Meta%20AI-Segment%20Anything%20ViT--B-0467DF)](https://segment-anything.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> End-to-end **Generative AI** pipeline combining **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5` & `runwayml/stable-diffusion-inpainting`) with Meta AI's **Segment Anything Model (SAM ViT-B)** for prompt-driven synthesis, two-stage latent refinement, zero-shot auto-mask inpainting, multi-directional outpainting, and an interactive **StudioAI** Streamlit web application.

---

## Background & Problem Statement

Standard text-to-image generation produces static images where users have little control over individual objects, framing, or canvas boundaries without regenerating the entire scene from scratch. Meanwhile, manual pixel-level masking is tedious for non-technical creators, and single-pass diffusion often forces a trade-off between prompt adherence, denoising steps, and VRAM efficiency.

This project addresses those gaps across two complementary deliverables:
1. **`Pipeline_GenAI_Project.ipynb`:** A systematic experimentation and editing pipeline covering diffusion hyperparameter tuning, scheduler benchmarking, two-stage latent refinement, manual vs. SAM zero-shot inpainting, and single-direction/zoom-out outpainting.
2. **`Streamlit_GenAI_Project.ipynb`:** A deployable interactive web app (**StudioAI**) with a freehand drawable canvas, batch selector, GPU memory flushing, and public tunneling via `pyngrok`.

---

## Objectives & Real-World Impact

- **Controlled Multimodal Synthesis:** Benchmark how `guidance_scale`, `num_inference_steps`, and sampling schedulers (`Euler A`, `DPM++`, `DDIM`) shape visual fidelity and reproducibility.
- **Object-Level & Boundary Editing:** Combine `SAM ViT-B` automatic segmentation and `StableDiffusionInpaintPipeline` for targeted object replacement and seamless 4-way canvas expansion (`Outpainting`).
- **Practical Industry Use Cases:**
  - **Creative Prototyping & Concept Art:** Accelerates visual ideation and scene expansion (zoom-out) for illustration, gaming, and film pre-production.
  - **E-Commerce & Advertising:** Enables automated object swapping and background extension on product visuals without full re-shoots.
  - **Accessible No-Code Editing:** Empowers non-programmers to generate, brush-mask, and outpaint images directly through the **StudioAI** browser UI.

---

## Repository Structure

```text
text-to-image-generation/
├── Pipeline_GenAI_Project.ipynb     # Core GenAI pipeline: Txt2Img, tuning, Img2Img refiner, Inpainting (Manual & SAM), Outpainting
├── Streamlit_GenAI_Project.ipynb    # Generates `logic.py` & `app.py` (StudioAI web app) + Colab ngrok deployment
├── requirements.txt                 # Pinned Python dependencies
└── README.md
```

---

## System Architecture & Pipeline Workflow

```text
Text Prompt + Negative Prompt + Seed
        │
        ▼
┌───────────────────────────────────────────────────┐
│  runwayml/stable-diffusion-v1-5 (FP16 + Slicing)  │  ← Schedulers: Euler A / DPM++ / DDIM
└───────────────────────────────────────────────────┘
        │
        ├──► [Img2Img Refiner]     Latent (denoising_end=0.8) ──► StableDiffusionImg2ImgPipeline (denoising_start=0.8)
        │
        ├──► [Auto-Mask Inpaint]   SAM ViT-B (25 segments) ──► Select target mask ──► StableDiffusionInpaintPipeline
        │
        ├──► [Outpainting]         Gaussian-blurred expanded canvas (+128px / 4-way +96px) ──► Inpaint outer mask
        │
        └──► [StudioAI Web App]    Streamlit UI (app.py + logic.py) + streamlit-drawable-canvas + ngrok (port 8501)
```

---

## Methodology & Implemented Modules

### 1. Core Pipeline (`Pipeline_GenAI_Project.ipynb`)
- **Text-to-Image & Hyperparameter Experiments** (base prompt: Martian astronaut digital illustration, `SEED = 222`):
  - **Simple vs. Advanced:** Default (`cfg=7.5`, `steps=50`) vs. tuned (`cfg=12.0`, `steps=80`).
  - **Guidance Scale Comparison (`steps=30`):** Evaluates `2.0` (loose/abstract), `7.5` (balanced), and `15.0` (strict prompt adherence with saturation risk).
  - **Inference Steps Comparison (`cfg=7.5`):** Evaluates `5`, `15`, `30`, and `50` steps.
  - **Batch Inference:** Generates a `2×2` grid (`num_images_per_prompt=4`) from a single prompt.
  - **Scheduler Comparison:** Compares `EulerAncestralDiscreteScheduler` (`Euler A`), `DPMSolverMultistepScheduler` (`DPM++`), and `DDIMScheduler` (`DDIM`).
- **Two-Stage Base + Refiner:** Stage 1 generates latents up to `denoising_end=0.8`; Stage 2 decodes/refines via `StableDiffusionImg2ImgPipeline` from `denoising_start=0.8` (`50` total steps).
- **Inpainting (`runwayml/stable-diffusion-inpainting`, `512×512`):**
  - *Manual Masking:* Draws a rectangular mask `[250, 200, 490, 430]` to insert a crashed satellite (`cfg=12.0`, `steps=60`).
  - *SAM Auto-Masking:* Loads `sam_vit_b_01ec64.pth` with `SamAutomaticMaskGenerator` (`points_per_side=16`, `pred_iou_thresh=0.88`, `stability_score_thresh=0.92`, `min_mask_region_area=500`) and isolates the `second_largest` segment to replace the astronaut with a metallic robot astronaut.
- **Outpainting & Iterative Zoom-Out:**
  - *Single-Direction:* Expands right edge by `+128px` over a Gaussian-blurred background (`radius=30`).
  - *Zoom-Out (4-Way):* Expands all 4 sides by `96px` (`radius=50` blur, multiples of 8) iteratively (`512×512` → `704×704` → `896×896` → `1088×1088`) and inpaints the surrounding landscape.

### 2. StudioAI Interactive App (`Streamlit_GenAI_Project.ipynb`)
Generates `logic.py` and `app.py` via `%%writefile`:
- **`logic.py`:** Cached FP16 model loader (`load_models_cached`), `flush_memory()` (`gc.collect()` + `torch.cuda.empty_cache()`), `set_scheduler()`, batch `generate_image()`, `run_inpainting()`, and `prepare_outpainting()` (`+128px` all sides).
- **`app.py`:** Two-tab interface (`✨ GENERATE` with `1–4` batch grid & image selector; `🛠️ EDIT` with `streamlit-drawable-canvas` freehand masking + debug mask expander and 1-click Zoom-Out mode).

---

## Experimental Findings & Results

- **Optimal Steps & CFG Trade-off:** `5–15` steps leave heavy noise and missing garment/terrain details; **`30` steps** provides the best quality-to-compute sweet spot (`50` steps yields marginal edge smoothing). Optimal `guidance_scale` falls in **`7–13`** (`2.0` drifts off-prompt; `15.0` introduces over-saturation artifacts).
- **Scheduler Characteristics:** `Euler A` produces expressive, organic textures; **`DPM++`** delivers the sharpest, cleanest convergence at 30 steps; **`DDIM`** offers smooth, highly reproducible deterministic outputs.
- **SAM Zero-Shot Segmentation:** `SamAutomaticMaskGenerator` successfully detected **25 distinct segments** on the generated `512×512` Mars image, cleanly isolating the foreground astronaut (`second_largest` mask) for targeted inpainting without manual brush drawing.
- **Outpainting & Deployment:** Both directional (`+128px`) and 4-way Zoom-Out (`512×512` → `704×704`) preserved scene lighting and Martian terrain continuity; the Streamlit app launched successfully on port `8501` via `pyngrok`.

---

## Tech Stack

| Category | Library | Pinned Version (`requirements.txt`) |
| :--- | :--- | :--- |
| Diffusion Pipelines | `diffusers`, `accelerate` | `0.27.2` / `0.30.0` *(notebook installs `>=0.30.0`)* |
| Text Encoder / Backend | `transformers` | `4.40.0` |
| Deep Learning & Attention | `torch`, `torchvision`, `xformers` | `2.3.0` / `0.18.0` / `0.0.26.post1` |
| Segmentation | `segment-anything` (SAM ViT-B) | `1.0` |
| Web UI & Tunneling | `streamlit`, `streamlit-drawable-canvas`, `pyngrok` | `1.35.0` / `0.8.0` / `7.2.0` |
| Image & Numerics | `Pillow`, `numpy`, `scipy`, `matplotlib`, `invisible-watermark` | `10.3.0` / `1.26.4` / `1.13.0` / `3.9.0` / `0.2.0` |

---

## Getting Started

### 1. Clone & Install Dependencies
> Requires **Python 3.9+** and a **CUDA-capable GPU** (minimum 8 GB VRAM recommended for FP16 Stable Diffusion + SAM).

```bash
git clone https://github.com/Bimzt/text-to-image-generation.git
cd text-to-image-generation
pip install -r requirements.txt
```

### 2. Run the Core Pipeline Notebook
Open `Pipeline_GenAI_Project.ipynb` in Google Colab (T4/L4 GPU) or Jupyter and run cells sequentially. Cell 1 automatically downloads `sam_vit_b_01ec64.pth` (~375 MB).

### 3. Launch the StudioAI Streamlit App (Colab + ngrok)
1. Open `Streamlit_GenAI_Project.ipynb` in Google Colab with GPU enabled.
2. Replace `auth_token` in the ngrok cell with your own token from [dashboard.ngrok.com](https://dashboard.ngrok.com):
   ```python
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   ```
3. Run all cells to write `logic.py` and `app.py`, start Streamlit on port `8501`, and print the public ngrok URL. Close active tunnels afterward with `ngrok.kill()`.
