"""
One-off preprocessing: walk the official DAiSEE folder layout and extract
1-fps frames for every clip into a flat cache directory, so training
doesn't need to decode video every epoch.

Expected input layout (official DAiSEE release):
    DAiSEE/
      DataSet/
        Train/<subject_id>/<clip_id>/<clip_id>.avi
        Validation/...
        Test/...
      Labels/
        TrainLabels.csv
        ValidationLabels.csv
        TestLabels.csv

Usage:
    python scripts/extract_daisee_frames.py \
        --daisee_root ./data/DAiSEE \
        --out_root ./data/DAiSEE_frames \
        --fps 1
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import extract_frames_at_fps  # noqa: E402


def find_clip_video(dataset_root: str, split_dir: str, clip_id_no_ext: str):
    pattern = os.path.join(dataset_root, split_dir, "**", clip_id_no_ext + ".*")
    matches = glob.glob(pattern, recursive=True)
    video_exts = (".avi", ".mp4", ".mov")
    matches = [m for m in matches if m.lower().endswith(video_exts)]
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daisee_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    args = parser.parse_args()

    dataset_root = os.path.join(args.daisee_root, "DataSet")

    for split in args.splits:
        print(f"[{split}] scanning videos...")
        pattern = os.path.join(dataset_root, split, "**", "*.avi")
        videos = glob.glob(pattern, recursive=True)
        if not videos:
            print(f"  WARNING: no .avi files found under {os.path.join(dataset_root, split)}. "
                  f"Check --daisee_root path / dataset layout.")
        for i, video_path in enumerate(videos):
            clip_id = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(args.out_root, split, clip_id)
            try:
                frames = extract_frames_at_fps(video_path, out_dir, target_fps=args.fps)
            except Exception as e:
                print(f"  FAILED on {video_path}: {e}")
                continue
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i + 1}/{len(videos)} clips processed")
        print(f"[{split}] done: {len(videos)} clips.")


if __name__ == "__main__":
    main()
