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

# 3) Review events (Streamlit skeleton)
streamlit run src/vap/review/app.py
```

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
  states.py             # state machine logic (drive/wait/walk)
  actions.py            # heuristic actions (grab/place/remove/load...)
  events.py             # event dataclasses and writers
  io.py                 # CSV/Parquet writers
  run.py                # CLI entry point
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

## License
MIT (adjust as needed).
