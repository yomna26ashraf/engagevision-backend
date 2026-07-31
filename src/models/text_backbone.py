"""
Text feature extractor (Section III-B): RoBERTa.

For missing/empty transcripts, the paper uses "global text mean
imputation" — we compute the mean RoBERTa embedding over the training
set once and reuse it whenever a transcript segment is empty.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast


class RoBERTaTextEncoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base", trainable: bool = False,
                 max_length: int = 128):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size  # 768 for roberta-base
        self.max_length = max_length
        self._global_mean: Optional[torch.Tensor] = None  # set via set_global_mean()

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def set_global_mean(self, mean_vec: torch.Tensor):
        """Call once after computing the mean embedding over the training
        split, to be used as the imputation value for missing transcripts."""
        self.register_buffer("_global_mean_buf", mean_vec, persistent=False)
        self._global_mean = mean_vec

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list of strings (batch). Empty strings are imputed with the
        global mean embedding if `set_global_mean` was called; otherwise
        they fall back to a zero vector with a warning-free no-op.
        """
        device = next(self.model.parameters()).device
        outputs = torch.zeros(len(texts), self.output_dim, device=device)

        non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        empty_idx = [i for i in range(len(texts)) if i not in non_empty_idx]

        if non_empty_idx:
            batch_texts = [texts[i] for i in non_empty_idx]
            enc = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(device)
            with torch.set_grad_enabled(self.model.training):
                out = self.model(**enc)
            # Mean-pool over tokens (masked) as the sentence representation.
            token_embeds = out.last_hidden_state  # (n, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            for j, i in enumerate(non_empty_idx):
                outputs[i] = pooled[j]

        if empty_idx and self._global_mean is not None:
            outputs[empty_idx] = self._global_mean.to(device)

        return outputs

    @torch.no_grad()
    def compute_global_mean(self, all_texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Utility to precompute the global mean embedding over a corpus of
        (non-empty) training transcripts. Call this once, then
        `set_global_mean(result)`."""
        non_empty = [t for t in all_texts if t and t.strip()]
        sums = torch.zeros(self.output_dim)
        count = 0
        for i in range(0, len(non_empty), batch_size):
            batch = non_empty[i:i + batch_size]
            embeds = self.forward(batch).detach().cpu()
            sums += embeds.sum(0)
            count += embeds.shape[0]
        return sums / max(count, 1)
