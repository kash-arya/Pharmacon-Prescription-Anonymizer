"""Pharmacon — Tiny per-writer fine-tune feasibility test (FINAL, verified)"""

# 1. Install deps (uncomment if fresh runtime)
# !pip install -q "transformers>=4.46.0" einops timm peft pillow jiwer
# !pip install -q -U "torchao>=0.16.0"

import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from peft import LoraConfig, get_peft_model
from jiwer import cer, wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Upload the 15 crops
from google.colab import files
uploaded = files.upload()
print("Uploaded:", len(uploaded), "files")

# 3. Label map
LABELS = {
    "train1 - 11.png": "once daily twice a day three times a day at bedtime as needed",
    "train(6) - 15.png": "mg mcg ml gm IU tsp tbsp drop puff",
    "train(2) - 17.png": "Tab. Paracetamol 500mg 1-1-1 PC",
    "train(3) - 18.png": "Tab. Amoxicillin 500mg 1-1-1 PC x 5 days",
    "train(7) - 25.png": "Tab. Metoprolol 50mg 1-0-1 PC",
    "train(9) - 29.png": "Cap. Omeprazole 20mg 1-0-0 AC",
    "train(8) - 38.png": "Cap. Doxycycline 100mg 1-0-1 PC x 7 days",
    "train(4) - 42.png": "Syp. Azithromycin 200mg/5ml 2 tsp 1-0-0 AC x 3 days",
    "train(10) - 44.png": "Tab. Folic Acid 5mg 1-0-0 PC",
    "train(5) - 46.png": "Cap. Tramadol 50mg SOS",
    "test(1) - 1.png": "Tab. Glimepiride 2mg 1-0-0 AC",
    "test(2) - 2.png": "Tab. Vildagliptin 50mg 1-0-1 PC",
    "test(3) - 22.png": "Cap. Fluconazole 150mg 1-0-0 PC",
    "test(4) - 24.png": "Tab. Cotrimoxazole 960mg 1-0-1 PC x 5 days",
    "test(5) - 25.png": "Syp. Amoxicillin 125mg/5ml 1 tsp 1-1-1 PC x 5 days",
}

TRAIN_FILES = [k for k in LABELS if k.startswith("train")]
TEST_FILES  = [k for k in LABELS if k.startswith("test")]

# 4. Load Florence-2 (compatibility patches FIRST)
PretrainedConfig.forced_bos_token_id = None
PreTrainedTokenizerBase.additional_special_tokens = property(
    lambda self: self.special_tokens_map.get("additional_special_tokens", [])
)

MODEL_ID = "microsoft/florence-2-large"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def fixed_get_imports(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True
    ).to(device)
base_model.eval()
print("Florence-2 loaded")

# 5. Prepare items — resize to square (Florence-2 requires 768x768)
def load_item(fname):
    img = Image.open(fname).convert("RGB").resize((768, 768))
    return img, LABELS[fname]

# 6. Fine-tune with LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config)
# LoRA wrapper lacks generation_config; copy it from the base model.
model.generation_config = base_model.generation_config
model.print_trainable_parameters()

EPOCHS = 20
LR = 1e-4
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

model.train()
for epoch in range(EPOCHS):
    total = 0.0
    for fname in TRAIN_FILES:
        img, text = load_item(fname)
        inputs = processor(text="<OCR>", images=img, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
        labels = processor.tokenizer(text=text, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = out.loss
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"epoch {epoch+1}/{EPOCHS} loss={total/len(TRAIN_FILES):.4f}")

model.eval()

# 7. Evaluate on held-out
def predict(img):
    inputs = processor(text="<OCR>", images=img, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                             max_new_tokens=64, num_beams=3, do_sample=False,
                             use_cache=False)
    text = processor.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task="<OCR>", image_size=img.size)
    return parsed.get("<OCR>", "").strip()

print("\n=== HELD-OUT RESULTS ===")
preds, truths = [], []
for fname in TEST_FILES:
    img, truth = load_item(fname)
    p = predict(img)
    preds.append(p); truths.append(truth)
    print(f"\nGT:   {truth}")
    print(f"PRED: {p}")

exact = np.mean([p.lower()==t.lower() for p,t in zip(preds, truths)])
print(f"\nExact match: {exact:.0%}")
print(f"CER: {np.mean([cer(t,p) for p,t in zip(preds,truths)]):.1%}")
print(f"WER: {np.mean([wer(t,p) for p,t in zip(preds,truths)]):.1%}")
