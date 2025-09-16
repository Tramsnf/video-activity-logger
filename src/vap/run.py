from __future__ import annotations
import argparse
from .config import load_config
from .pipeline import analyze_video

def main():
    ap = argparse.ArgumentParser(description="Video Activity Logger")
    ap.add_argument("--config", required=True, help="Path to pipeline YAML")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = analyze_video(cfg, args.video, output_dir=args.out)
    if result.event_csv_path:
        print(f"Wrote {len(result.events)} events → {result.event_csv_path}")
    else:
        print(f"Wrote {len(result.events)} events")

if __name__ == "__main__":
    main()
