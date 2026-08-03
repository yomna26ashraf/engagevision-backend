# Deploying EngageVision AI (portfolio setup)

## Overview
- **Frontend** (TanStack Start app) → deploy via Lovable's Publish button
  (easiest, since it was built there), or manually to Cloudflare Pages
  (it's already configured for that) or Vercel.
- **Backend** (FastAPI + PyTorch) → needs a host that runs arbitrary
  Python/Docker, since it's not a static site. Render or Railway both
  have simple free/cheap tiers good enough for occasional demo traffic.

## Redeploying with ONNX (recommended — much smaller memory footprint)

If you exported and validated an ONNX model (see the main README's "Exporting
to ONNX" section) and hit OOM crashes on Render's free tier with the full
PyTorch backend, switch to the ONNX-backed service instead:

1. Upload `checkpoints/daisee_mlatte.onnx` to the **same Hugging Face model
   repo** you used for the `.pt` checkpoint (Files and versions → Add file →
   Upload files), and copy its direct download link the same way.
2. On Render → your service → **Environment**, add/update:
   - `MLATTE_USE_ONNX` = `1`
   - `MLATTE_ONNX_URL` = the direct download link from step 1
   - `MLATTE_LOW_MEMORY` = `1` (still helps — int8-quantizes the ONNX graph too)
   - You can remove `MLATTE_CHECKPOINT_URL` now; the ONNX path doesn't need it.
3. Trigger a redeploy (push a commit, or Manual Deploy).

This avoids loading full PyTorch + torchvision + the dual-ResNet-50 model into
memory at all — onnxruntime's footprint for the same model is substantially
smaller, which is the whole point of this path.

## Backend: deploying to Render (or Railway — steps are nearly identical)

1. **Get your trained checkpoint somewhere downloadable.** Don't commit
   `checkpoints/daisee_mlatte_best.pt` to git (it's a large binary). Instead:
   - Upload it to a Hugging Face Hub model repo (free, simplest — gives you
     a direct download URL), OR
   - Upload it to any file host that gives a direct download link (S3,
     a GitHub Release asset, Google Drive with a direct-download link).

2. **Push this repo to GitHub** (checkpoints/ excluded via .gitignore —
   already set up).

3. **On Render:** New → Web Service → connect your GitHub repo.
   - Root directory: repo root (Dockerfile lives at `backend/Dockerfile`,
     but it expects to be built from the repo root — set the Dockerfile
     path to `backend/Dockerfile` and keep build context as the repo root).
   - Environment variables:
     - `MLATTE_CHECKPOINT_URL` = the direct download link from step 1
       (the backend downloads it once on first startup).
     - `ALLOWED_ORIGINS` = your deployed frontend's exact URL, e.g.
       `https://your-site.pages.dev`
     - `MLATTE_LOW_MEMORY` = `1` — **important on free/512MB-class tiers.**
       Enables three memory-saving tricks with no accuracy impact:
       chunked CNN inference (2 frames at a time instead of all 10 at
       once), int8 dynamic quantization of the VAE/regression Linear
       layers, and a single CPU thread. See `backend/model_service.py`
       for details. Combined with `opencv-python-headless` (already in
       requirements.txt) this meaningfully lowers the peak memory a
       single prediction needs — but a dual-ResNet-50 model is still
       heavy; if you still hit OOM on a 512MB tier after this, the
       realistic next steps are a host with more RAM (Standard tier,
       Google Cloud Run, etc.) or accepting a local-only demo for now.
   - Plan: the free/cheapest CPU plan is fine to start; predictions will
     take a few seconds each (CPU inference), which is acceptable for a
     portfolio demo you're clicking through live, not a high-traffic app.

4. Once deployed, note the backend's public URL (e.g.
   `https://engagevision-api.onrender.com`).

## Frontend: pointing it at the deployed backend

Set the frontend's environment variable before building/publishing:
```
VITE_API_BASE_URL=https://engagevision-api.onrender.com
```
- **Via Lovable:** add this in the project's environment variable settings,
  then Publish.
- **Via Cloudflare Pages/Vercel manually:** add it in the project's
  environment variables dashboard, then trigger a rebuild.

## Cold starts (free tiers sleep)
Free tiers on Render/Railway spin the service down after inactivity — the
first request after idling can take 30-60s while it wakes up and loads the
model. For a portfolio demo this is usually fine (mention it in a small
note near the demo, e.g. "first prediction may take ~30s to wake the
server"), or upgrade to a plan that doesn't sleep if it bothers you.

## Local Docker test (before deploying anywhere)
```bash
cd mlatte
docker build -f backend/Dockerfile -t engagevision-backend .
docker run -p 8000:8000 -e MLATTE_CHECKPOINT_URL="<your-url>" engagevision-backend
curl http://localhost:8000/api/health
```
