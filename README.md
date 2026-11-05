# Qwen-Image-2.1 Turbo Uncensored — Google Colab

One-click Colab setup for Qwen-Image-2.1 **Turbo** (7B, text-to-image + instruction editing)
with the **heretic (abliterated) Qwen3-VL-8B text encoder**, running on a free T4 16 GB via GGUF.

**Open in Colab:**
https://colab.research.google.com/github/herbert-kara/qwen-image-21-colab/blob/main/Qwen_Image_2.1_Uncensored_Colab.ipynb

Set `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.

## What the notebook installs

| Piece | File | Size |
|---|---|---:|
| DiT (turbo, 8 steps, cfg 1) | `qwen-image-2.1-turbo-Q4_K_M.gguf` — `abenzerps/Qwen-Image-2.1-Turbo-Quantized` | 4.05 GB |
| Text encoder (heretic) | `qwen3vl_8b_heretic-Q4_K_M.gguf` — `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF` | 5.03 GB |
| Vision tower for the encoder | `mmproj-qwen3vl_8b_heretic-f16.gguf` | 1.16 GB |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |
| LoRA (optional) | `NSFW Qwen Lora.safetensors` — `f23gg/NSFW-LORA-Qwen-Image-2.1` | 0.16 GB |

≈ 11 GB total — fits a free T4. For A100/L4 set `QUANT = "Q8_0"` in cell 3.

## Software

- `comfyanonymous/ComfyUI` (latest master) in `/content/ComfyUI`
- `city96/ComfyUI-GGUF` (GGUF unet/text-encoder loaders)
- `pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE` — **required**: ComfyUI-GGUF does not load the
  Qwen3-VL vision tower on its own (`got input of size [1, 512, 12288]`), and it also adds
  Qwen-Image-2.1 DiT architecture detection for GGUF files.

## Usage notes

- Loaders: `Unet Loader (GGUF)` + `CLIPLoader (GGUF)` with type `qwen_image` + `VAELoader`.
- **8 steps, cfg 1.0** (turbo, no CFG). More steps are wasted.
- Keep the `mmproj` file next to the encoder GGUF and **do not rename either file** — the patch
  finds the vision tower by name.
- Text rendered inside the image: two chained `KSamplerAdvanced` (first at cfg 1.0, second at
  cfg 3.0 from step ~15 to 25) exactly as described in the pottokao model card.
- Out of memory on T4: add `--lowvram` to the command in cell 4.
- Cell 4 prints the GUI URL from Colab's own proxy (`google.colab.kernel.proxyPort`), so no
  gradio/cloudflared tunnel is involved. If that URL 404s, add `--share` to the same command
  and take the `gradio.live` link from the cell output instead.

## If Colab refuses a GPU (`Cannot connect to GPU backend — usage limits`)

That message is account-level free-tier quota, not a bug you can code around. In order of cheapness:

1. **Switch Google account** in Colab (top-right avatar) — a fresh account has a fresh quota. Instant.
2. **Wait** — the free quota is dynamic and typically frees up within a few hours to ~24 h.
3. **Pay-as-you-go compute units / Colab Pro** — the only guaranteed path.

**Use the waiting time:** the notebook's cell 3 stages the 11 GB of weights into **Google Drive**
(`MyDrive/qwen21-models`) and symlinks them into ComfyUI, so it runs on a **CPU runtime** too
(CPU runtime has no quota). Once a GPU window opens, only cell 1 and cell 4 are needed — up in
about 2 minutes instead of 20. Drive has 15 GB free; the cache is ~11 GB.

## Notes / caveats

- Qwen-Image-2.1 weights are under the **Qwen research license** — commercial use needs a separate
  license from Alibaba. `Qwen/Qwen-Image-Edit-2511` is Apache-2.0 if that matters (it is 20B and
  does not fit a T4 comfortably).
- Editing here is **instruction-driven only, no mask/inpainting**.
