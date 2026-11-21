"""LLM wrapper (HF transformers, 4-bit), retriever, and a CPU mock for plumbing tests."""
import gc
import json
import re

import numpy as np

from .data import NUM_UNIT_RE


class LLM:
    def __init__(self, name, load_4bit=True, max_new_tokens=700):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch, self.name, self.max_new_tokens = torch, name, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"device_map": "auto"}
        if torch.cuda.is_available() and load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.model.eval()

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, batch_of_messages, batch_size=4):
        torch = self.torch
        prompts = [self.render(m) for m in batch_of_messages]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths together
        outs = [None] * len(prompts)
        self.model.eval()
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id, use_cache=True)
            for k, i in enumerate(idx):
                outs[i] = self.tok.decode(gen[k, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class MockLLM:
    """No-GPU stand-in: regex-extracts number+unit pairs and answers in the requested format.
    Only for testing that the pipeline runs end to end; its scores mean nothing."""
    name = "mock"
    supports_lora = False

    def generate(self, batch_of_messages, batch_size=4):
        outs = []
        for msgs in batch_of_messages:
            text = msgs[-1]["content"].split("Text:\n", 1)[-1]
            recs = regex_records(text)
            if "NONE" in msgs[0]["content"]:
                outs.append("\n".join(f"{r['material']} | {r['property']} | {r['value']} | {r['unit']}" for r in recs) or "NONE")
            else:
                outs.append(json.dumps([{**r, "confidence": 0.5} for r in recs]))
        return outs

    def close(self):
        pass


def regex_records(text):
    """Rule-based baseline: every number+unit mention becomes a record (no material/property)."""
    recs = []
    for m in NUM_UNIT_RE.finditer(text):
        s = m.group(0)
        mm = re.match(r"(\d+(?:\.\d+)?(?:\s?±\s?\d+(?:\.\d+)?)?)\s*(.*)", s, flags=re.S)
        sent = next((x for x in re.split(r"(?<=[.!?])\s+", text) if s in x), "")
        recs.append({"material": "", "property": "", "value": mm.group(1).strip(),
                     "unit": mm.group(2).strip(), "evidence": sent, "confidence": None})
    return recs


class Retriever:
    """Nearest-neighbour demo retrieval. Uses sentence-transformers if installed, else TF-IDF."""

    def __init__(self, use_dense=True):
        self.model = None
        if use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.model = None

    def fit(self, texts):
        if self.model is not None:
            self.emb = self.model.encode(texts, normalize_embeddings=True)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(sublinear_tf=True).fit(texts)
            self.emb = self.vec.transform(texts)
        return self

    def topk(self, query, k=3):
        if self.model is not None:
            sims = self.emb @ self.model.encode([query], normalize_embeddings=True)[0]
        else:
            sims = (self.emb @ self.vec.transform([query]).T).toarray().ravel()
        return list(np.argsort(-sims)[:k])
