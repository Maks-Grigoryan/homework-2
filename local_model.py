from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalInstructionModel:
    def __init__(self, model_id: str, cache_dir: Path) -> None:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()

    def complete(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def rank(
        self, system: str, user: str, candidates: list[str]
    ) -> tuple[str, dict[str, float]]:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        candidate_ids = [
            self.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            for candidate in candidates
        ]
        prompt_length = prompt_ids.shape[1]
        max_length = prompt_length + max(ids.shape[0] for ids in candidate_ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = torch.full((len(candidates), max_length), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, ids in enumerate(candidate_ids):
            combined = torch.cat((prompt_ids[0], ids))
            input_ids[row, : combined.shape[0]] = combined
            attention[row, : combined.shape[0]] = 1
        scores: dict[str, float] = {}
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            for row, (candidate, ids) in enumerate(zip(candidates, candidate_ids)):
                start = prompt_length - 1
                token_logits = logits[row : row + 1, start : start + ids.shape[0], :]
                log_probs = torch.log_softmax(token_logits, dim=-1)
                chosen = log_probs.gather(
                    2, ids.reshape(1, -1, 1)
                ).squeeze(-1)
                scores[candidate] = float(chosen.mean().item())
        return max(candidates, key=scores.__getitem__), scores


def dialogue_text(dialogue: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{index}. {turn['role'].upper()}: {turn['text']}"
        for index, turn in enumerate(dialogue, start=1)
    )
