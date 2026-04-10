"""
VaultMind Fine-Tuning Pipeline
Phase 4 -- Custom model training from user feedback data.

After enough thumbs-up feedback accumulates, this module fine-tunes
a small language model on the user's specific patterns:
  - Terminology and jargon
  - Preferred citation formats
  - Common query types
  - Writing style and tone

Uses Unsloth + LoRA (same pipeline as AIR Blackbox) for efficient
local fine-tuning. Exports to GGUF for Ollama.

Pipeline:
  1. Export training data from feedback_loop.py
  2. Format into instruction/input/output triplets
  3. Fine-tune with Unsloth + LoRA (4-bit quantized)
  4. Export to GGUF
  5. Register with Ollama as a custom model
  6. Update routing to prefer the fine-tuned model

Requirements (installed on demand):
  pip install unsloth transformers datasets peft

Storage: ~/.vaultmind/models/
No cloud. Training happens 100% locally.
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from typing import Optional


# ── Config ────────────────────────────────────────────────────

MODELS_DIR = os.path.expanduser("~/.vaultmind/models")
TRAINING_DIR = os.path.join(MODELS_DIR, "training")
OUTPUT_DIR = os.path.join(MODELS_DIR, "output")
OLLAMA_MODELS_DIR = os.path.expanduser("~/.ollama/models")

# Fine-tuning defaults
DEFAULT_BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 2e-4
DEFAULT_MAX_SEQ_LEN = 2048
MIN_TRAINING_SAMPLES = 50  # Need at least this many before training is worth it


# ── Pipeline Status ───────────────────────────────────────────

class PipelineStatus:
    IDLE = "idle"
    PREPARING = "preparing_data"
    TRAINING = "training"
    EXPORTING = "exporting_gguf"
    REGISTERING = "registering_ollama"
    COMPLETE = "complete"
    FAILED = "failed"


_current_status = PipelineStatus.IDLE
_status_detail = ""


def get_status() -> dict:
    """Get the current pipeline status."""
    return {
        "status": _current_status,
        "detail": _status_detail,
        "models_dir": MODELS_DIR,
        "has_trained_model": _has_trained_model(),
    }


def _has_trained_model() -> bool:
    """Check if a fine-tuned model exists."""
    gguf_path = os.path.join(OUTPUT_DIR, "vaultmind-custom.gguf")
    return os.path.exists(gguf_path)


def _set_status(status: str, detail: str = ""):
    global _current_status, _status_detail
    _current_status = status
    _status_detail = detail
    print(f"[FineTune] {status}: {detail}")


# ── Step 1: Prepare Training Data ─────────────────────────────

def prepare_training_data(feedback_data: list = None) -> dict:
    """Prepare training data from feedback entries.

    Args:
        feedback_data: List of dicts with instruction/input/output keys.
            If None, imports from feedback_loop.py.

    Returns:
        dict with data_path, sample_count, and readiness info.
    """
    _set_status(PipelineStatus.PREPARING, "Loading feedback data")

    if feedback_data is None:
        try:
            from feedback_loop import export_training_data
            feedback_data = export_training_data(min_rating=1)
        except Exception as e:
            _set_status(PipelineStatus.FAILED, f"Could not load feedback data: {e}")
            return {"error": str(e)}

    sample_count = len(feedback_data)

    if sample_count < MIN_TRAINING_SAMPLES:
        _set_status(PipelineStatus.IDLE, f"Not enough data yet ({sample_count}/{MIN_TRAINING_SAMPLES})")
        return {
            "ready": False,
            "sample_count": sample_count,
            "needed": MIN_TRAINING_SAMPLES,
            "message": f"Need {MIN_TRAINING_SAMPLES - sample_count} more positive feedback entries before training.",
        }

    # Format data for Unsloth
    os.makedirs(TRAINING_DIR, exist_ok=True)
    formatted = []
    for entry in feedback_data:
        formatted.append({
            "instruction": entry.get("instruction", "Answer the question using the provided context."),
            "input": entry.get("input", ""),
            "output": entry.get("output", ""),
        })

    data_path = os.path.join(TRAINING_DIR, "training_data.json")
    with open(data_path, "w") as f:
        json.dump(formatted, f, indent=2)

    _set_status(PipelineStatus.IDLE, f"Data ready: {sample_count} samples")
    return {
        "ready": True,
        "sample_count": sample_count,
        "data_path": data_path,
        "message": f"Training data prepared: {sample_count} samples ready.",
    }


# ── Step 2: Generate Training Script ─────────────────────────

def _generate_training_script(
    data_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_rank: int = DEFAULT_LORA_RANK,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> str:
    """Generate a Python training script for Unsloth + LoRA.

    Returns the script file path.
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)
    script_path = os.path.join(TRAINING_DIR, "train.py")

    script = f'''"""Auto-generated VaultMind fine-tuning script."""
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load training data
with open("{data_path}") as f:
    raw_data = json.load(f)

# Format for Alpaca template
def format_prompt(sample):
    if sample["input"]:
        return f"""### Instruction:\\n{{sample["instruction"]}}\\n\\n### Input:\\n{{sample["input"]}}\\n\\n### Response:\\n{{sample["output"]}}"""
    return f"""### Instruction:\\n{{sample["instruction"]}}\\n\\n### Response:\\n{{sample["output"]}}"""

texts = [format_prompt(s) for s in raw_data]
dataset = Dataset.from_dict({{"text": texts}})

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length={max_seq_len},
    dtype=None,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_rank},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha={lora_rank},
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={max_seq_len},
    args=TrainingArguments(
        output_dir="{OUTPUT_DIR}/checkpoints",
        per_device_train_batch_size={batch_size},
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs={epochs},
        learning_rate={learning_rate},
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        seed=42,
    ),
)

trainer.train()

# Save LoRA adapter
model.save_pretrained("{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained("{OUTPUT_DIR}/lora_adapter")

# Export to GGUF for Ollama
model.save_pretrained_gguf("{OUTPUT_DIR}", tokenizer, quantization_method="q4_k_m")

print("Training complete! GGUF exported to {OUTPUT_DIR}")
'''

    with open(script_path, "w") as f:
        f.write(script)

    return script_path


# ── Step 3: Run Training ──────────────────────────────────────

def run_training(
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
) -> dict:
    """Run the full fine-tuning pipeline.

    This is a blocking operation that can take 10-60 minutes
    depending on dataset size and hardware.
    """
    # Step 1: Prepare data
    data_result = prepare_training_data()
    if not data_result.get("ready"):
        return data_result

    data_path = data_result["data_path"]

    # Step 2: Generate script
    _set_status(PipelineStatus.TRAINING, "Generating training script")
    script_path = _generate_training_script(
        data_path=data_path,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Step 3: Run training
    _set_status(PipelineStatus.TRAINING, "Fine-tuning in progress (this takes a while)")
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True, text=True, timeout=3600,
            cwd=TRAINING_DIR,
        )
        if result.returncode != 0:
            _set_status(PipelineStatus.FAILED, f"Training failed: {result.stderr[:500]}")
            return {"error": "Training failed", "stderr": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        _set_status(PipelineStatus.FAILED, "Training timed out (1 hour limit)")
        return {"error": "Training timed out"}
    except FileNotFoundError:
        _set_status(PipelineStatus.FAILED, "Python3 not found or Unsloth not installed")
        return {
            "error": "Dependencies not installed",
            "install_cmd": "pip install unsloth transformers datasets peft trl",
        }

    # Step 4: Register with Ollama
    register_result = register_with_ollama()
    if "error" in register_result:
        return register_result

    _set_status(PipelineStatus.COMPLETE, "Fine-tuned model ready!")
    return {
        "status": "complete",
        "model_name": "vaultmind-custom",
        "message": "Fine-tuned model registered with Ollama as 'vaultmind-custom'",
    }


# ── Step 4: Register with Ollama ──────────────────────────────

def register_with_ollama(
    model_name: str = "vaultmind-custom",
    gguf_filename: str = None,
) -> dict:
    """Register the fine-tuned GGUF model with Ollama.

    Creates a Modelfile and runs `ollama create`.
    """
    _set_status(PipelineStatus.REGISTERING, "Registering with Ollama")

    # Find the GGUF file
    if gguf_filename is None:
        gguf_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".gguf")]
        if not gguf_files:
            _set_status(PipelineStatus.FAILED, "No GGUF file found in output directory")
            return {"error": "No GGUF file found. Run training first."}
        gguf_filename = gguf_files[0]

    gguf_path = os.path.join(OUTPUT_DIR, gguf_filename)

    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0
PARAMETER num_predict 2048

SYSTEM \"\"\"You are VaultMind, a personal AI assistant fine-tuned on the user's documents and interaction patterns. You answer questions using ONLY the provided context. Never fabricate information. Cite sources when possible.\"\"\"
"""

    modelfile_path = os.path.join(OUTPUT_DIR, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # Register with Ollama
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            _set_status(PipelineStatus.FAILED, f"Ollama registration failed: {result.stderr[:300]}")
            return {"error": f"Ollama create failed: {result.stderr[:300]}"}
    except FileNotFoundError:
        _set_status(PipelineStatus.FAILED, "Ollama not found")
        return {"error": "Ollama not installed or not in PATH"}
    except subprocess.TimeoutExpired:
        _set_status(PipelineStatus.FAILED, "Ollama registration timed out")
        return {"error": "Ollama create timed out"}

    _set_status(PipelineStatus.COMPLETE, f"Model '{model_name}' registered successfully")
    return {
        "status": "registered",
        "model_name": model_name,
        "gguf_path": gguf_path,
        "message": f"Model '{model_name}' is now available in Ollama",
    }


# ── Readiness Check ───────────────────────────────────────────

def check_readiness() -> dict:
    """Check if the system is ready for fine-tuning.

    Returns a checklist of requirements and their status.
    """
    checks = {}

    # Check feedback data
    try:
        from feedback_loop import export_training_data
        data = export_training_data(min_rating=1)
        checks["training_data"] = {
            "ready": len(data) >= MIN_TRAINING_SAMPLES,
            "count": len(data),
            "needed": MIN_TRAINING_SAMPLES,
        }
    except Exception:
        checks["training_data"] = {"ready": False, "count": 0, "needed": MIN_TRAINING_SAMPLES}

    # Check Unsloth
    try:
        import unsloth
        checks["unsloth"] = {"ready": True, "version": getattr(unsloth, "__version__", "unknown")}
    except ImportError:
        checks["unsloth"] = {"ready": False, "install": "pip install unsloth"}

    # Check Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        checks["ollama"] = {"ready": result.returncode == 0}
    except Exception:
        checks["ollama"] = {"ready": False, "install": "https://ollama.ai/download"}

    # Check GPU
    try:
        import torch
        checks["gpu"] = {
            "ready": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        }
    except ImportError:
        checks["gpu"] = {"ready": False, "note": "PyTorch not installed"}

    # Check existing model
    checks["existing_model"] = {"exists": _has_trained_model()}

    # Overall readiness
    all_ready = all(
        checks[k].get("ready", False)
        for k in ["training_data", "ollama"]
    )

    return {
        "ready": all_ready,
        "checks": checks,
        "recommendation": _get_recommendation(checks),
    }


def _get_recommendation(checks: dict) -> str:
    """Generate a human-readable recommendation."""
    data = checks.get("training_data", {})
    if not data.get("ready"):
        count = data.get("count", 0)
        needed = data.get("needed", MIN_TRAINING_SAMPLES)
        return f"Keep using VaultMind and rating responses. Need {needed - count} more positive ratings before training is useful."

    if not checks.get("unsloth", {}).get("ready"):
        return "Install Unsloth for local fine-tuning: pip install unsloth"

    if not checks.get("gpu", {}).get("ready"):
        return "GPU recommended for fine-tuning. Training on CPU will be very slow (hours instead of minutes)."

    return "Ready to fine-tune! Run the training pipeline when you have 15-20 minutes."


# ── Training History ──────────────────────────────────────────

def get_training_history() -> list:
    """Get the history of training runs."""
    history_path = os.path.join(MODELS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    return []


def _log_training_run(result: dict):
    """Log a training run to history."""
    history = get_training_history()
    history.append({
        "timestamp": datetime.utcnow().isoformat(),
        **result,
    })
    os.makedirs(MODELS_DIR, exist_ok=True)
    history_path = os.path.join(MODELS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
