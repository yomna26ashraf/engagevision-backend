# M-LATTE — Reimplementation

Reimplementation of **"Multimodal Latent Temporal Modeling for Continuous
Engagement Assessment in Online Education"** (Xie et al., IEEE TLT, 2026),
scoped to the **DAiSEE** dataset (visual-only path) first, with the full
trimodal architecture (ViCEF + TrendCycleVAE) implemented and unit-tested
so it's ready for RoomReader/CMU-MOSI once you have access to those.

## 1. What's implemented

| Paper component | File | Status |
|---|---|---|
| Dual-branch visual encoder (Emotion + Behavior ResNet-50) | `src/models/visual_backbone.py` | ✅ |
| Audio encoder (VGGish) | `src/models/audio_backbone.py` | ✅ (needs `torch.hub` access once) |
| Text encoder (RoBERTa) | `src/models/text_backbone.py` | ✅ |
| ViCEF cross-modal fusion (Eqs. 6-10) | `src/models/vicef.py` | ✅ |
| FFT trend/cycle decomposition (Algorithm 1) | `src/models/fft_decomposition.py` | ✅ |
| TrendCycleVAE (dual-branch VAE) | `src/models/trend_cycle_vae.py` | ✅ |
| Regression head (Eq. 13) | `src/models/mlatte.py` | ✅ |
| Loss (Eqs. 14-18) | `src/losses.py` | ✅ |
| DAiSEE data pipeline | `src/data/daisee_dataset.py`, `scripts/extract_daisee_frames.py` | ✅ |
| Training loop (DAiSEE) | `scripts/train_daisee.py` | ✅ |
| Visual-branch pretraining (Table III) | `scripts/pretrain_visual_branch.py` | ✅ |
| RoomReader / CMU-MOSI trimodal training | `src/models/pipelines.py::TrimodalPipeline` | Skeleton only — see §6 |

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**First, sanity-check the architecture with synthetic tensors (no data needed):**

```bash
pytest tests/test_shapes.py -v
```

If this passes, every tensor shape through ViCEF → FFT decomposition →
TrendCycleVAE → regression head is wired correctly, before you invest time
in downloading data.

On your RTX 2000 Ada (16GB): the defaults in `configs/config.yaml` (batch
32, 4 transformer layers, d_model 256) should fit with mixed precision on.
If you hit OOM, drop `train.batch_size` to 8–16 and raise
`train.grad_accum_steps` proportionally (effective batch size stays ~32).

## 3. Getting the data

### DAiSEE (required — this is the dataset we're targeting)
Request access at the official DAiSEE page (a request form, typically
approved quickly) and place it as:

```
data/DAiSEE/
  DataSet/{Train,Validation,Test}/<subject>/<clip>/<clip>.avi
  Labels/{TrainLabels,ValidationLabels,TestLabels}.csv
```

Then extract 1-fps frames once (this is the slow step, run it overnight if needed):

```bash
python scripts/extract_daisee_frames.py \
    --daisee_root ./data/DAiSEE \
    --out_root ./data/DAiSEE_frames --fps 1
```

### RAF-DB / StudentEngagementDataset (for visual-branch pretraining, Table III)
Both also require a request form from their respective authors. **If you'd
rather not wait**, accessible substitutes that work with the same
`ImageFolder` pretraining script:
- Emotion branch: **FER2013** (Kaggle, instant download) — restructure
  into `train/<class>/*.jpg`, `val/<class>/*.jpg`.
- Behavior branch: no perfect public substitute exists; document this as a
  deviation (see §5) or annotate a small custom set from DAiSEE's own
  training clips (screen-viewing / writing / distracted) if you want a
  closer match.

```bash
python scripts/pretrain_visual_branch.py --branch emotion \
    --data_root ./data/FER2013 --num_classes 7 \
    --out_path ./checkpoints/emotion_branch.pt

python scripts/pretrain_visual_branch.py --branch behavior \
    --data_root ./data/YourBehaviorSubstitute --num_classes 3 \
    --out_path ./checkpoints/behavior_branch.pt
```

You *can* skip this step and train end-to-end from ImageNet init only —
the pipeline will warn you and still run — but expect lower accuracy than
the paper, since the paper's whole point is that domain pretraining
(RAF-DB / StudentEngagementDataset) gives the visual branch a head start.

## 4. Training on DAiSEE

```bash
python scripts/train_daisee.py \
    --config configs/config.yaml \
    --emotion_ckpt ./checkpoints/emotion_branch.pt \
    --behavior_ckpt ./checkpoints/behavior_branch.pt
```

This prints per-epoch train/val MSE, MAE, and classification accuracy
(predictions bucketed to the nearest DAiSEE level — see §5), then reports
final test-set numbers to compare against **the paper's 61.37% engagement
accuracy on DAiSEE (Table VI)**.

## 5. Known deviations from the paper (be upfront about these when reporting results)

1. **DAiSEE regression vs. classification**: the paper's Data-Processing
   section says DAiSEE labels are mapped to continuous values and trained
   with MSE, but Section IV-C-2 reports an *accuracy* number. We train the
   continuous regression head (matching the paper's core framework) and
   derive accuracy by snapping predictions to the nearest of
   `{0, 0.25, 0.5, 1.0}` — a reasonable reading of an ambiguous point in
   the paper, but a reading nonetheless.
2. **Pretraining data substitutes**: RAF-DB / StudentEngagementDataset are
   gated; FER2013 (emotion) is a common, close substitute, but there's no
   equally close public substitute for the 3-class behavior branch.
   Expect this to be the single biggest source of any accuracy gap vs.
   61.37%.
3. **VGGish / audio branch**: irrelevant for DAiSEE (no audio track), but
   implemented and unit-tested for when you move to RoomReader/CMU-MOSI.
4. **Window/segment granularity**: the paper's 32-second optimal window
   (Table XII) was tuned on RoomReader; DAiSEE clips are fixed 10-second
   segments, so we use `clip_seconds: 10` for that dataset instead, as the
   paper itself does.

## 6. Extending to RoomReader / CMU-MOSI (full trimodal path)

`MLATTEFull` and `ViCEF` are complete and unit-tested — what's missing is
dataset-specific glue code:
- **CMU-MOSI**: raw video/audio/transcripts are available from CMU
  MultiComp; you'll need to align them into fixed-length windows (see
  `src/data/preprocessing.py` for the frame-extraction pattern to mirror
  for audio chunks/transcript spans) and run them through
  `VGGishAudioEncoder` / `RoBERTaTextEncoder` before `MLATTEFull`.
- **RoomReader**: requires a data-use agreement with the dataset authors;
  once granted, the same windowing pattern applies (32s windows per
  Table XII).

`src/models/pipelines.py::TrimodalPipeline` is a thin end-to-end wrapper
you can adapt once you have a dataset class for either.

## 7. Roadmap: addressing the paper's own stated limitations (Phase 2)

Once you've matched (or understood any gap to) the paper's numbers on
DAiSEE, the paper's Conclusion section (V) lists four open problems worth
tackling next:

1. **Missing-modality robustness** — replace global-mean imputation
   (`RoBERTaTextEncoder.set_global_mean`) with modality dropout during
   training, or a learned imputation network.
2. **Short-term/fine-grained attention shifts** — add a local-attention
   branch operating on short sub-windows, fused alongside the existing
   trend/cycle decomposition.
3. **Real-time deployment efficiency** — swap ResNet-50 branches for a
   lighter backbone (MobileNetV3/EfficientNet) or distill the trained
   dual-branch encoder.
4. **Personalization** — add a learned per-student embedding, conditioning
   the regression head (Eq. 13) on it.

## 8. Exporting to ONNX (for memory-constrained deployment)

If your deployment host is too small for full PyTorch (e.g. a 512MB free
tier), export the trained model to ONNX first:

```bash
# 1. Confirm the matmul-based FFT (needed for ONNX export) matches
#    torch.fft exactly, before trusting anything built on it:
pytest tests/test_onnx_fft_equivalence.py -v

# 2. Export
python scripts/export_onnx.py \
    --checkpoint ./checkpoints/daisee_mlatte_best.pt \
    --out ./checkpoints/daisee_mlatte.onnx

# 3. Validate the export matches the original PyTorch model's predictions
#    on real random inputs — do not skip this step:
pip install onnxruntime
python scripts/validate_onnx.py \
    --checkpoint ./checkpoints/daisee_mlatte_best.pt \
    --onnx ./checkpoints/daisee_mlatte.onnx
```

Then run the backend with `MLATTE_USE_ONNX=1` (see `backend/onnx_model_service.py`)
to serve through `onnxruntime` instead of full PyTorch — a much smaller
memory footprint for the model itself. Note: preprocessing still uses
`torch`/`torchvision` under the hood (see that file's docstring for why),
so this reduces but doesn't eliminate the PyTorch dependency.

Two correctness notes baked into this export path:
- The VAE's reparameterization trick now uses the **mean** (deterministic)
  at inference instead of sampling — fixes a real bug where the same clip
  could previously get a slightly different score on every request.
- FFT is computed via precomputed DFT/IDFT matrices (`src/models/fft_matrix.py`)
  instead of `torch.fft`, which has poor/no ONNX Runtime support for our
  fixed, non-power-of-two clip length (10). This is mathematically exact
  for a fixed sequence length, not an approximation — verified by
  `tests/test_onnx_fft_equivalence.py`.

