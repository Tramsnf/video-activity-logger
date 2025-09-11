#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Batch process videos with VAP pipeline")
    ap.add_argument("--config", required=True, help="Path to pipeline YAML")
    ap.add_argument("--videos_dir", required=True, help="Directory containing videos")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--exts", nargs="*", default=[".mp4", ".mov", ".avi", ".mkv"], help="Video extensions to include")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (uses subprocesses)")
    args = ap.parse_args()

    vdir = Path(args.videos_dir)
    if not vdir.exists():
        print(f"Videos dir not found: {vdir}")
        sys.exit(1)

    videos = [p for p in vdir.rglob("*") if p.suffix.lower() in set(e.lower() for e in args.exts)]
    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos. Starting...")

    def run_one(path: Path):
        cmd = [sys.executable, "-m", "vap.run", "--config", args.config, "--video", str(path), "--out", args.out]
        print(" [2mRUN[0m", " ".join(cmd))
        return subprocess.call(cmd)

    if args.workers <= 1:
        for p in videos:
            rc = run_one(p)
            if rc != 0:
                print(f"Failed on {p} with code {rc}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, p): p for p in videos}
            for f in as_completed(futs):
                p = futs[f]
                try:
                    rc = f.result()
                except Exception as e:
                    print(f"Error on {p}: {e}")
                else:
                    if rc != 0:
                        print(f"Failed on {p} with code {rc}")


if __name__ == "__main__":
    main()

