"""
Dual-branch visual encoder (Section III-B of the paper).

Two ResNet-50 backbones:
  - Emotion branch ("AffectNet" in the paper's naming): pretrained for
    7-way facial-emotion classification (paper uses RAF-DB).
  - Behavior branch ("BehaviorNet"): pretrained for 3-way student-behavior
    classification (screen-viewing / writing / distraction), paper uses
    the StudentEngagementDataset.

At inference time for the main model, we discard each branch's classifier
head and concatenate the penultimate-layer (2048-d each) activations to
form F_V (4096-d), exactly as described in the paper.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class _ResNet50Classifier(nn.Module):
    """A ResNet-50 with a replaceable classification head.

    Used standalone during branch pretraining (Table III), and then reused
    (with the head stripped) as a frozen/fine-tuned feature extractor.
    """

    def __init__(self, num_classes: int, pretrained_imagenet: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_imagenet else None
        backbone = tv_models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        # Keep everything except the final FC layer as the "trunk".
        self.trunk = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 2048, 1, 1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.trunk(x).flatten(1)  # (B, 2048)
        if return_features:
            return feats
        return self.classifier(feats)


class EmotionBranch(_ResNet50Classifier):
    """AffectNet-style branch: 7 basic emotions (neutral, happy, sad,
    surprise, fear, disgust, anger) as in RAF-DB."""

    def __init__(self, num_classes: int = 7, pretrained_imagenet: bool = True):
        super().__init__(num_classes=num_classes, pretrained_imagenet=pretrained_imagenet)


class BehaviorBranch(_ResNet50Classifier):
    """BehaviorNet-style branch: 3 student behaviors (screen-viewing,
    writing, distraction) as in StudentEngagementDataset."""

    def __init__(self, num_classes: int = 3, pretrained_imagenet: bool = True):
        super().__init__(num_classes=num_classes, pretrained_imagenet=pretrained_imagenet)


class DualBranchVisualEncoder(nn.Module):
    """Combines the two pretrained branches into the fused visual feature F_V.

    Usage:
        emo = EmotionBranch(); load pretrained weights
        beh = BehaviorBranch(); load pretrained weights
        encoder = DualBranchVisualEncoder(emo, beh)
        F_V = encoder(frames)   # (B, T, 4096) or (B, 4096) depending on input rank
    """

    def __init__(self, emotion_branch: EmotionBranch, behavior_branch: BehaviorBranch,
                 freeze_backbones: bool = False, freeze_emotion: bool = None,
                 freeze_behavior: bool = None, inference_chunk_size: Optional[int] = None):
        """
        freeze_backbones: convenience flag applied to BOTH branches equally.
        freeze_emotion / freeze_behavior: per-branch overrides (e.g. freeze
            a well-pretrained emotion branch while leaving an
            ImageNet-only behavior branch trainable so it can still adapt
            during downstream training). If set, these take precedence
            over `freeze_backbones` for their respective branch.
        inference_chunk_size: if set, frames are pushed through each CNN
            branch in small groups (e.g. 2 at a time) instead of all at
            once, trading a bit of speed for a much lower peak activation
            memory footprint — useful on memory-constrained CPU hosts
            (e.g. a 512MB deployment tier). Does not change the result,
            only how much intermediate memory is held at once. Leave as
            None during training (default) for full throughput.
        """
        super().__init__()
        self.emotion_branch = emotion_branch
        self.behavior_branch = behavior_branch
        self.output_dim = emotion_branch.feature_dim + behavior_branch.feature_dim  # 4096
        self.inference_chunk_size = inference_chunk_size

        freeze_emotion = freeze_backbones if freeze_emotion is None else freeze_emotion
        freeze_behavior = freeze_backbones if freeze_behavior is None else freeze_behavior
        self._freeze_emotion = freeze_emotion
        self._freeze_behavior = freeze_behavior

        if freeze_emotion:
            for p in self.emotion_branch.parameters():
                p.requires_grad = False
        if freeze_behavior:
            for p in self.behavior_branch.parameters():
                p.requires_grad = False

    def _branch_forward(self, branch: nn.Module, x: torch.Tensor, frozen: bool) -> torch.Tensor:
        """Runs a branch under no_grad() when it's frozen, so PyTorch never
        allocates activation memory for a backward pass that will never
        happen through this branch's own weights — a meaningful memory
        saving on 16GB-class GPUs with dual ResNet-50 branches. Also
        chunks the batch dimension when `self.inference_chunk_size` is set
        (see __init__ docstring)."""
        chunk = self.inference_chunk_size
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        with ctx:
            if chunk is None or x.shape[0] <= chunk:
                return branch(x, return_features=True)
            outputs = [branch(x[i:i + chunk], return_features=True) for i in range(0, x.shape[0], chunk)]
            return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) for a single frame, or (B, T, C, H, W) for a clip.
        Returns F_V of shape (B, 4096) or (B, T, 4096).
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            emo_feats = self._branch_forward(self.emotion_branch, x, self._freeze_emotion)
            beh_feats = self._branch_forward(self.behavior_branch, x, self._freeze_behavior)
            fused = torch.cat([emo_feats, beh_feats], dim=-1)
            return fused.view(b, t, -1)
        elif x.dim() == 4:
            emo_feats = self._branch_forward(self.emotion_branch, x, self._freeze_emotion)
            beh_feats = self._branch_forward(self.behavior_branch, x, self._freeze_behavior)
            return torch.cat([emo_feats, beh_feats], dim=-1)
        else:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")
