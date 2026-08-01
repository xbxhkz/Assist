# image_training_sidecar/train_sdxl_lora.py
"""SDXL LoRA training sidecar. Runs INSIDE the Py3.11 CUDA venv (never
imported by the Py3.14 app). Reads a JSON config describing a saved image
dataset (image + caption pairs), trains a rank-N LoRA against the SDXL
UNet using the precompute-then-offload technique the feasibility spike
proved fits this hardware (see docs/superpowers/specs/2026-07-31-image-
lora-training-engine-design.md): every image/caption pair is VAE- and
text-encoder-encoded ONCE up front while those modules are resident, then
the text encoders + VAE are moved off the GPU entirely so only the UNet
(+ LoRA + 8-bit optimizer state) stays resident for the training loop.
Streams JSON-line progress to stdout and writes the finished LoRA
straight to the path the manager resolved in imagemodels' loras_dir()
registry -- no conversion step, unlike text-LoRA training's GGUF step.

Progress protocol (one JSON object per line on stdout):
  {"event":"start","total_steps":N}
  {"event":"step","step":N,"loss":F,"vram_gb":F}
  {"event":"done","lora_path":str,"peak_vram_gb":F}
  {"event":"error","message":str,"trace":str}
"""
import argparse
import json
import os
import random
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    # stdout is the JSON-progress channel back to the app; force UTF-8 with
    # a non-raising error handler so a non-ASCII message can never crash
    # emit() (Windows Py3.11 stdout defaults to the locale encoding).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True)
        args = ap.parse_args()
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Suppress huggingface_hub/diffusers download+loading progress bars --
        # they redraw with \r on stdout and would corrupt the JSON progress
        # channel above, exactly the cross-seam bug the text-LoRA trainer's
        # whole-branch review caught (its fix: disable_tqdm=True on Trainer;
        # this script has no Trainer, so the equivalent is disabling the
        # library-wide progress bars before anything else prints).
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()

        import torch
        from diffusers import StableDiffusionXLPipeline, DDPMScheduler
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        import bitsandbytes as bnb
        from PIL import Image

        images = cfg["images"]  # [{"image": path, "caption": text}, ...]
        base_model = cfg["base_model"]
        rank = int(cfg.get("rank", 4))
        lora_alpha = int(cfg.get("lora_alpha", 4))
        learning_rate = float(cfg.get("learning_rate", 1e-4))
        steps = int(cfg.get("steps", 1000))
        resolution = int(cfg.get("resolution", 1024))
        lora_path = cfg["lora_path"]
        run_dir = cfg["run_dir"]
        os.makedirs(run_dir, exist_ok=True)

        device = "cuda"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # --- Phase 1: precompute every image's latents + text embedding ONCE,
        # while the text encoders + VAE are still resident. ---
        pipe.vae.to(device, dtype=torch.float32)
        pipe.text_encoder.to(device, dtype=torch.float16)
        pipe.text_encoder_2.to(device, dtype=torch.float16)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)

        examples = []
        for item in images:
            img = Image.open(item["image"]).convert("RGB")
            pixel_values = pipe.image_processor.preprocess(img, height=resolution, width=resolution)
            pixel_values = pixel_values.to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = (latents * pipe.vae.config.scaling_factor).to(dtype=torch.float16)
                (prompt_embeds, _, pooled_prompt_embeds, _) = pipe.encode_prompt(
                    prompt=item.get("caption") or "", device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
            examples.append((latents.detach(), prompt_embeds.detach(), pooled_prompt_embeds.detach()))

        # --- Phase 2: offload text encoders + VAE, keep only the UNet resident. ---
        pipe.vae.to("cpu")
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pipe.unet.to(device, dtype=torch.float16)
        pipe.unet.requires_grad_(False)
        pipe.unet.enable_gradient_checkpointing()

        lora_config = LoraConfig(r=rank, lora_alpha=lora_alpha,
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        pipe.unet.add_adapter(lora_config)
        lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=learning_rate)

        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            device=device, dtype=torch.float16)

        emit({"event": "start", "total_steps": steps})
        for step in range(1, steps + 1):
            latents, prompt_embeds, pooled_prompt_embeds = random.choice(examples)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (1,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            model_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs).sample

            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            emit({"event": "step", "step": step, "loss": round(float(loss.item()), 4), "vram_gb": vram})

        unet_lora_state_dict = get_peft_model_state_dict(pipe.unet)
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=run_dir, unet_lora_layers=unet_lora_state_dict)
        produced = os.path.join(run_dir, "pytorch_lora_weights.safetensors")
        os.replace(produced, lora_path)

        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        emit({"event": "done", "lora_path": lora_path, "peak_vram_gb": peak})
    except (Exception, SystemExit) as e:  # noqa: BLE001
        import traceback
        try:
            emit({"event": "error", "message": f"{e}", "trace": traceback.format_exc()[-1500:]})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
