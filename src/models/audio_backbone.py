"""
Audio feature extractor (Section III-B): VGGish pretrained on AudioSet.

VGGish is not part of torchvision/torchaudio's pretrained model zoo, so we
load it via torch.hub (harritaylor/torchvggish), which mirrors Google's
original AudioSet-pretrained checkpoint. This produces 128-d embeddings
per ~0.96s audio frame; we mean-pool (or keep per-frame) to match the
window granularity used elsewhere in the pipeline.

If torch.hub is unavailable in your environment (e.g. offline cluster),
download the checkpoint manually and point VGGISH_LOCAL_DIR to it — see
the README for instructions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VGGishAudioEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, trainable: bool = False):
        super().__init__()
        try:
            self.model = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=pretrained)
        except Exception as e:  # pragma: no cover - depends on network access
            raise RuntimeError(
                "Could not load VGGish via torch.hub. Download the checkpoint "
                "manually (see README 'Audio backbone' section) and adapt this "
                "loader to load it from a local path."
            ) from e
        self.model.postprocess = False  # keep raw embeddings, no PCA/whitening quantization
        self.output_dim = 128

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, waveform_path_or_array, sample_rate: int = 16000) -> torch.Tensor:
        """
        Accepts either a filepath (str) or a 1-D numpy/torch waveform array.
        Returns embeddings of shape (num_frames, 128); mean-pool externally
        if you need a single vector per window.
        """
        embeddings = self.model.forward(waveform_path_or_array, sample_rate)
        return embeddings

    def encode_window(self, waveform_path_or_array, sample_rate: int = 16000) -> torch.Tensor:
        """Convenience wrapper: mean-pool frame embeddings into a single
        window-level feature vector F_A of shape (128,)."""
        emb = self.forward(waveform_path_or_array, sample_rate)
        if emb.numel() == 0:
            return torch.zeros(self.output_dim)
        return emb.mean(dim=0)
