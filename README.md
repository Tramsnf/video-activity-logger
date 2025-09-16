# Video Activity Logger (VAP)

Detect, track, and timestamp warehouse activities (e.g., forklift **DRIVE**, **WAIT**, **GRAB_SKID**, **PLACE_SKID**) from video.
Outputs canonical event logs (CSV/Parquet) driven by a single taxonomy and thresholds.

## Quickstart
```bash
# 1) Create a venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run pipeline (mock detector by default)
python -m vap.run --config configs/pilot.yaml --video /path/to/video.mp4 --out outputs

# 3) Launch the Studio UI
streamlit run src/vap/web/app.py
# or: python scripts/run_studio.py

# (Optional) Review an events CSV
streamlit run src/vap/review/app.py
```

## New Features (Robust Pipeline)
- Trackers: configurable `iou` (default), optional `bytetrack` and `ocsort` backends with graceful fallback.
- Per-actor thresholds: forklift vs human speeds and debounced transitions.
- State cleanup: merge small gaps and drop too-short intervals.
- Actions: heuristic GRAB_SKID/PLACE_SKID; configurable distance/frame parameters.
- Zones: optional polygons (`configs/zones.yaml`) with ZONE_ENTER/ZONE_EXIT events.
- Batch: `scripts/batch_run.py` to process a directory of videos.
- Feature export: `scripts/export_features.py` generates per-actor time series for training TCNs.
- Evaluation: `scripts/eval_events.py` computes event-level F1 and timestamp MAE.
- Model export/training helpers: `scripts/export_model.py`, `scripts/train_yolo.sh`.

### Config knobs
- `detect`: `backend`, `model_path`, `classes`, `min_conf`, `min_box_area`, `imgsz`, `batch_size`, `device`, `fp16`
- `thresholds`: `fl_speed_*`, `hu_speed_*`, `debounce_frames`, `min_state_dur_s`, `merge_gap_s`, action_*
- Optional: `roi_mask_path`, `zones_path`

### Zones example
Add to `configs/pilot.yaml`:
```yaml
zones_path: configs/zones.yaml
```

### Training
- Prepare a YOLO `data.yaml` with `train/val/test` splits and names `[person, forklift, pallet]`.
- Run: `bash scripts/train_yolo.sh configs/data_forklift.yaml yolov8n.pt 100 forklift-v1`
- Use the resulting weights in `configs/pilot.yaml` as `detect.model_path`.

> Tip: Start with the mock detector to validate end-to-end flow. Switch to YOLO by editing `configs/pilot.yaml`.

## Repo layout
```
configs/
  taxonomy.yaml         # canonical activities (states/events/markers)
  thresholds.yaml       # speed/debounce thresholds per actor type
  mapping.csv           # your raw→canonical mapping
  pilot.yaml            # end-to-end pipeline config
src/vap/
  __init__.py
  config.py             # config models & loader
  taxonomy.py           # taxonomy loader & validation
  ingest.py             # video reader & fps normalization
  detect.py             # detector interface (mock/yolo)
  track.py              # multi-object tracking wrapper
  pipeline.py           # reusable pipeline runner & annotation helpers
  states.py             # state machine logic (drive/wait/walk)
  actions.py            # heuristic actions (grab/place/remove/load...)
  events.py             # event dataclasses and writers
  io.py                 # CSV/Parquet writers
  run.py                # CLI entry point
  web/app.py            # interactive Streamlit studio for running videos
  review/app.py         # Streamlit skeleton to scrub outputs
tests/
  test_taxonomy.py      # sanity check taxonomy schema
```

## Switch detectors
- **mock**: no detections (pipeline still runs); good for wiring
- **yolo**: Ultralytics model (`yolov8n.pt` or your fine-tuned weights)
  - Configure in `configs/pilot.yaml`:
    ```yaml
    detect:
      backend: yolo
      model_path: yolov8n.pt
      classes: [person, forklift, pallet]
    ```

## Outputs
Events CSV with columns:
`video_id, actor_id, actor_type, activity, start_time_s, end_time_s, duration_s, confidence, source_camera, attributes`

## Performance tips
- Install `av` (`pip install av`) to unlock the PyAV video reader, which is much faster than the OpenCV fallback on longer clips.
- Keep `detect.batch_size` as high as your GPU/CPU can handle in the Streamlit sidebar to increase throughput.
- Trim `imgsz` and confidence thresholds for quick experiments, then restore to production values for full-accuracy runs.

## License
MIT (adjust as needed).
