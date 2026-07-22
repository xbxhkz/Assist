"""QLoRA training sidecar. Runs INSIDE the Python 3.11 CUDA venv (never imported
by the Py3.14 app). Reads a JSON config, fine-tunes a base model with a 4-bit
QLoRA, streams JSON-line progress to stdout, and saves a LoRA adapter.

Progress protocol (one JSON object per line on stdout):
  {"event":"start","model":..,"total_steps":..}
  {"event":"step","step":N,"loss":..,"vram_gb":..}
  {"event":"done","output_dir":..,"peak_vram_gb":..}
  {"event":"error","message":..}
"""
import argparse
import json
import os
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling, TrainerCallback)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        # dataset is normalized to [{"text": ...}] by the app before launch
        with open(cfg["dataset_path"], "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        model_id = cfg["base_model"]
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        max_len = int(cfg.get("max_seq_length", 512))

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
                          lora_dropout=float(cfg.get("lora_dropout", 0.05)), bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora)
        model.config.use_cache = False

        ds = Dataset.from_list(rows).map(
            lambda ex: tok(ex["text"], truncation=True, max_length=max_len, padding="max_length"),
            remove_columns=["text"])
        collator = DataCollatorForLanguageModeling(tok, mlm=False)

        steps = cfg.get("steps")
        epochs = cfg.get("epochs")
        targs = dict(output_dir=out_dir, per_device_train_batch_size=int(cfg.get("batch_size", 1)),
                     gradient_accumulation_steps=1, learning_rate=float(cfg.get("learning_rate", 2e-4)),
                     logging_steps=1, gradient_checkpointing=True, fp16=True,
                     report_to=[], save_strategy="no")
        if steps:
            targs["max_steps"] = int(steps)
        else:
            targs["num_train_epochs"] = float(epochs or 1)
        targ = TrainingArguments(**targs)

        total = int(steps) if steps else None
        emit({"event": "start", "model": model_id, "total_steps": total})

        class Cb(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    v = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0
                    emit({"event": "step", "step": int(state.global_step),
                          "loss": round(float(logs["loss"]), 4), "vram_gb": v})

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(model=model, args=targ, train_dataset=ds,
                          data_collator=collator, callbacks=[Cb()])
        trainer.train()

        trainer.model.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0
        emit({"event": "done", "output_dir": out_dir, "peak_vram_gb": peak})
    except Exception as e:  # noqa: BLE001
        import traceback
        emit({"event": "error", "message": f"{e}", "trace": traceback.format_exc()[-1500:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
