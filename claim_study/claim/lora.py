"""QLoRA fine-tuning of the extraction model (plain PyTorch loop; no trl dependency)."""
import json
import math
import random

from .prompts import build_messages
from .utils import autofill_evidence


def target_json(doc):
    out = []
    for r in doc["records"]:
        item = {k: r.get(k, "") for k in ["material", "property", "value", "unit"]}
        item["evidence"] = autofill_evidence(doc["text"], r)
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def _encode(tok, doc, max_len):
    prompt = tok.apply_chat_template(build_messages("lora_json", doc["text"]), tokenize=False,
                                     add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(target_json(doc) + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


def train_lora(llm, train_docs, epochs=3, lr=2e-4, grad_accum=8, max_len=1536, r=16, seed=42, log=print):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok, model = llm.tok, llm.model
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False

    data = [_encode(tok, d, max_len) for d in train_docs]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = max(1, math.ceil(len(data) * epochs / grad_accum))
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    rng = random.Random(seed)
    model.train()
    step, run_loss = 0, 0.0
    for ep in range(epochs):
        rng.shuffle(data)
        for i, (ids, labels) in enumerate(data):
            x = torch.tensor([ids], device=model.device)
            y = torch.tensor([labels], device=model.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                loss = model(input_ids=x, labels=y).loss / grad_accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
            if (i + 1) % grad_accum == 0 or i == len(data) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 5 == 0 or step == total:
                    log(f"  [lora] epoch {ep + 1}/{epochs} step {step}/{total} loss {run_loss / 5:.4f}")
                    run_loss = 0.0
    model.eval()
    model.config.use_cache = True
    llm.model = model
    return llm
