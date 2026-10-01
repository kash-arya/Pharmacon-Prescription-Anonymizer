"""
Writer adaptation for TrOCR-small.

Implements the protocol's adaptation recipe:
  - Base model: microsoft/trocr-small-handwritten (62M).
  - LoRA rank 8 on query+value of the LAST TWO visual-encoder attention blocks.
  - Decoder frozen (so the model can't memorise medicine names).

Verified working in Colab (see trOCR_smoke_test.ipynb). Key gotchas handled:
  - TrOCR's tokenizer must be loaded directly (XLMRobertaTokenizer), not via
    AutoTokenizer, to avoid the TokenizersBackend error on newer transformers.
  - torchao version check in PEFT must be bypassed (Colab ships incompatible
    0.10.0).
  - The DeiT encoder attention modules are named `q_proj`/`v_proj`, not
    `query`/`value`.
"""

import re

import torch
from transformers import (AutoImageProcessor, XLMRobertaTokenizer,
                          TrOCRProcessor, VisionEncoderDecoderModel)
from peft import LoraConfig, get_peft_model

MODEL_ID = "microsoft/trocr-small-handwritten"


def _bypass_torchao_check():
    """Disable PEFT's torchao availability check (Colab ships incompatible 0.10.0)."""
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass


def load_trOCR(model_id=MODEL_ID):
    """Load TrOCR-small processor + model. Returns (processor, model)."""
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_id)
    processor = TrOCRProcessor(image_processor=image_processor,
                               tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    return processor, model


def _encoder_attn_modules(model):
    """Return [(layer_index, module_name)] for encoder query/value projections."""
    found = []
    for name, _ in model.named_modules():
        if not name.startswith("encoder."):
            continue
        if any(name.endswith(k) for k in [".q_proj", ".v_proj",
                                          ".query", ".value"]):
            m = re.search(r"layer[s]?\.(\d+)", name)
            if m:
                found.append((int(m.group(1)), name))
    return sorted(found, key=lambda x: (x[0], x[1]))


def make_writer_adapter(model, rank=8):
    """
    Wrap `model` with LoRA (rank 8) on the last two encoder blocks'
    query+value projections, decoder frozen. Returns (peft_model, targets).
    """
    _bypass_torchao_check()

    attn = _encoder_attn_modules(model)
    if not attn:
        raise RuntimeError("Could not detect encoder query/value modules.")

    max_layer = max(idx for idx, _ in attn)
    last_two = (max_layer - 1, max_layer)
    targets = [name for idx, name in attn if idx in last_two]

    assert len(targets) == 4, \
        f"Expected 4 target modules, got {len(targets)}: {targets}"

    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.15,
        target_modules=targets, bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, targets


def train_adapter(peft_model, processor, images, labels,
                  val_images=None, val_labels=None,
                  epochs=35, lr=1e-4, patience=4, device=None):
    """Fine-tune on (images, labels) with optional early stopping on val set."""
    import copy

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)

    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)

    history = []
    best_val_loss = float("inf")
    best_weights = None
    p_counter = 0

    for epoch in range(1, epochs + 1):
        peft_model.train()
        total = 0.0
        for img, text in zip(images, labels):
            pixel_values = processor(images=img,
                                     return_tensors="pt").pixel_values.to(device)
            label_ids = processor.tokenizer(
                text=text, return_tensors="pt").input_ids.to(device)
            out = peft_model(pixel_values=pixel_values, labels=label_ids)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        epoch_loss = total / max(len(images), 1)
        history.append(epoch_loss)

        if val_images and val_labels:
            peft_model.eval()
            val_total = 0.0
            with torch.no_grad():
                for v_img, v_text in zip(val_images, val_labels):
                    v_pv = processor(images=v_img, return_tensors="pt").pixel_values.to(device)
                    v_ids = processor.tokenizer(text=v_text, return_tensors="pt").input_ids.to(device)
                    val_total += peft_model(pixel_values=v_pv, labels=v_ids).loss.item()
            val_loss = val_total / max(len(val_images), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(peft_model.state_dict())
                p_counter = 0
            else:
                p_counter += 1
                if p_counter >= patience:
                    break

    if best_weights:
        peft_model.load_state_dict(best_weights)

    peft_model.eval()
    return history


def predict(peft_model, processor, image, device=None):
    """Return recognised text for one image using beam search decoding."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    peft_model.eval()
    pixel_values = processor(images=image,
                             return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = peft_model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
