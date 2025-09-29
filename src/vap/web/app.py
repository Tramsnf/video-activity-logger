from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

# Ensure the package root (src/) is importable when running via `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vap.config import load_config, PipelineConfig
from vap.pipeline import PipelineResult, analyze_video
from vap.events import COLUMNS as EVENT_COLUMNS


st.set_page_config(page_title="Video Activity Logger Studio", layout="wide")
st.title("Video Activity Logger — Live Studio")


def _available_configs() -> Dict[str, Path]:
    configs_dir = Path("configs")
    paths = list(configs_dir.glob("*.yaml"))
    paths.sort(key=lambda p: (0 if p.name == "pilot.yaml" else 1, p.name.lower()))
    return {p.name: p for p in paths}


def _available_videos() -> Dict[str, Path]:
    workspace = Path(".")
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    return {p.name: p for p in sorted(workspace.glob("*")) if p.suffix.lower() in extensions}


def _available_models() -> Dict[str, Path]:
    workspace = Path(".").resolve()
    seen: Dict[Path, str] = {}
    candidates: Dict[str, Path] = {}
    search_roots = [workspace, workspace / "models", workspace / "weights"]
    patterns = ("*.pt", "*.onnx")

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for item in root.glob(pattern):
                if not item.is_file():
                    continue
                resolved = item.resolve()
                if resolved in seen:
                    continue
                try:
                    label = str(resolved.relative_to(workspace))
                except ValueError:
                    label = resolved.name
                seen[resolved] = label
                candidates[label] = resolved

    return dict(sorted(candidates.items(), key=lambda kv: kv[0].lower()))


def _save_uploaded_file(upload) -> str:
    suffix = Path(upload.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getvalue())
    tmp.close()
    return tmp.name


def _format_events(result: PipelineResult) -> pd.DataFrame:
    if not result.events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(result.events)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if "attributes" in df.columns:
        def _fmt_attributes(val):
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val or {}, ensure_ascii=False)
            except Exception:
                return "{}"
        df["attributes"] = df["attributes"].apply(_fmt_attributes)
    return df[EVENT_COLUMNS]


def _run_analysis(
    config: PipelineConfig,
    video_path: str,
    detection_params: Dict[str, object],
    max_frames: Optional[int],
    generate_preview: bool,
    frame_callback=None,
) -> PipelineResult:
    progress_bar = st.progress(0.0, text="Starting analysis…")

    def _progress_cb(frame_idx: int, total: Optional[int]) -> None:
        if total and total > 0:
            ratio = min(1.0, frame_idx / total)
            text = f"Processing frame {frame_idx:,}/{total:,}"
        else:
            ratio = 0.0 if frame_idx <= 0 else 1.0
            text = f"Processed {frame_idx:,} frames"
        progress_bar.progress(ratio, text=text)

    overrides = {k: v for k, v in detection_params.items() if v not in (None, "")}

    result = analyze_video(
        config,
        video_path,
        detection_overrides=overrides,
        create_annotated_video=generate_preview,
        progress_callback=_progress_cb,
        frame_callback=frame_callback,
        max_frames=max_frames if max_frames and max_frames > 0 else None,
    )
    progress_bar.progress(1.0, text="Analysis complete")
    return result


def main() -> None:
    st.session_state.setdefault("vap_auto_ready", False)

    configs = _available_configs()
    config_names = list(configs.keys())
    if not config_names:
        st.error("No pipeline configs found in `configs/`. Add a YAML config to continue.")
        return

    with st.sidebar:
        st.header("Pipeline")
        config_name = st.selectbox("Config file", config_names, index=0)
        config = load_config(str(configs[config_name]))

        st.header("Detection")
        model_default = config.detect.model_path or ""
        available_models = _available_models()
        custom_label = "Custom path"

        selected_label = custom_label
        workspace_root = Path(".").resolve()
        if model_default:
            expanded_default = Path(model_default).expanduser()
            resolved_default = expanded_default.resolve(strict=False)
            for label, path in available_models.items():
                if path == resolved_default:
                    selected_label = label
                    break
            else:
                if expanded_default.is_file():
                    try:
                        label = str(resolved_default.relative_to(workspace_root))
                    except ValueError:
                        label = str(resolved_default)
                    available_models[label] = resolved_default
                    selected_label = label
        elif available_models:
            selected_label = next(iter(available_models.keys()))

        model_options = list(available_models.keys())
        model_options.sort(key=lambda s: s.lower())
        if selected_label not in model_options and selected_label != custom_label:
            model_options.append(selected_label)
        if custom_label not in model_options:
            model_options.append(custom_label)

        default_index = model_options.index(selected_label) if selected_label in model_options else model_options.index(custom_label)
        model_choice = st.selectbox("Model weights", model_options, index=default_index if default_index >= 0 else 0)

        if model_choice == custom_label:
            model_path = st.text_input(
                "Custom model path",
                value=model_default,
                help="Path to the YOLO model weights to load.",
            )
        else:
            model_path = str(available_models[model_choice])
        min_conf = st.slider("Min confidence", 0.01, 1.0, float(config.detect.min_conf), 0.01)
        min_box_area = st.number_input("Min box area", min_value=1, value=int(config.detect.min_box_area), step=10)
        imgsz = st.number_input("Image size", min_value=320, max_value=1920, value=int(config.detect.imgsz), step=32)
        iou = st.slider("NMS IoU", 0.05, 0.95, float(config.detect.iou), 0.01)
        agnostic_nms = st.checkbox("Class-agnostic NMS", value=bool(config.detect.agnostic_nms))
        max_det = st.number_input("Max detections", min_value=1, value=int(config.detect.max_det), step=50)
        batch_size = st.slider("Batch size", 1, 16, int(config.detect.batch_size or 1))
        fp16 = st.checkbox("Enable FP16", value=bool(config.detect.fp16))
        device = st.text_input("Device", value=config.detect.device or "", help="Set to GPU index (e.g. 0) or 'cpu'. Leave blank for default.")

        st.header("Runtime")
        preview_frames = st.number_input("Max frames (0 = full video)", min_value=0, value=0, help="Limit processing for quick previews.")
        generate_preview = st.checkbox("Render annotated preview", value=True)
        live_preview = st.checkbox("Show live preview", value=True)
        live_stride = st.slider("Live preview stride", 1, 10, value=1, help="Update the live frame every N frames to save resources.")

    video_sources = _available_videos()
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    existing_choice = None
    if video_sources:
        existing_choice = st.selectbox("Or pick a local video", ["-"] + list(video_sources.keys()))

    video_path: Optional[str] = None
    if uploaded_video is not None:
        video_path = _save_uploaded_file(uploaded_video)
        st.info(f"Uploaded video saved to temporary file: {video_path}")
    elif existing_choice and existing_choice != "-":
        video_path = str(video_sources[existing_choice])

    detection_params = {
        "model_path": model_path,
        "min_conf": float(min_conf),
        "min_box_area": int(min_box_area),
        "imgsz": int(imgsz),
        "iou": float(iou),
        "agnostic_nms": bool(agnostic_nms),
        "max_det": int(max_det),
        "batch_size": int(batch_size),
        "fp16": bool(fp16),
        "device": device.strip() or None,
    }

    control_signature = {
        "config": config_name,
        "video": video_path,
        "detection": detection_params,
        "preview_frames": int(preview_frames),
        "generate_preview": bool(generate_preview),
        "live_preview": bool(live_preview),
        "live_stride": int(live_stride),
    }

    prev_signature = st.session_state.get("vap_last_controls")
    auto_trigger = False
    if (
        video_path
        and prev_signature is not None
        and prev_signature != control_signature
        and st.session_state.get("vap_last_video") == video_path
        and st.session_state.get("vap_auto_ready")
    ):
        auto_trigger = True

    st.session_state["vap_last_controls"] = control_signature

    run_requested = st.button("Run analysis", type="primary", disabled=video_path is None)
    if not run_requested and auto_trigger and video_path:
        run_requested = True

    live_frame_placeholder = st.empty()

    if run_requested and video_path:
        with st.spinner("Running pipeline…"):
            def _live_frame_cb(frame_idx: int, frame):
                if not live_preview:
                    return
                if frame_idx <= 0:
                    return
                if live_stride > 1 and frame_idx % live_stride != 0:
                    return
                try:
                    rgb = frame[:, :, ::-1]
                except Exception:
                    rgb = frame
                live_frame_placeholder.image(
                    rgb,
                    caption=f"Frame {frame_idx:,}",
                    width="stretch",
                )
            st.session_state["vap_auto_ready"] = True
            st.session_state["vap_last_video"] = video_path
            try:
                result = _run_analysis(
                    config,
                    video_path,
                    detection_params,
                    int(preview_frames),
                    generate_preview,
                    frame_callback=_live_frame_cb,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.session_state["vap_auto_ready"] = False
                return
        st.session_state["vap_last_result"] = result
        st.session_state["vap_last_events"] = _format_events(result)

    if "vap_last_result" not in st.session_state:
        st.info("Upload or select a video, then press Run analysis once. After the first run, tweaks auto-apply for the same video.")
        return

    result: PipelineResult = st.session_state["vap_last_result"]
    df: pd.DataFrame = st.session_state.get("vap_last_events", pd.DataFrame(columns=EVENT_COLUMNS))

    total_frames = getattr(result, "total_frames", None)
    if total_frames:
        st.success(
            f"Processed {result.frame_count:,}/{total_frames:,} frames · {len(result.events)} events logged."
        )
    else:
        st.success(
            f"Processed {result.frame_count:,} frames · {len(result.events)} events logged."
        )

    stats = result.frame_stats
    if stats:
        stats_df = pd.DataFrame([s.__dict__ for s in stats])
        avg_det = stats_df["num_detections"].mean()
        avg_trk = stats_df["num_tracks"].mean()
        col1, col2, col3 = st.columns(3)
        if total_frames:
            col1.metric("Frames processed", f"{result.frame_count:,}/{total_frames:,}")
        else:
            col1.metric("Frames processed", f"{result.frame_count:,}")
        col2.metric("Events", f"{len(result.events):,}")
        col3.metric("Avg det/track", f"{avg_det:.2f} / {avg_trk:.2f}")
        st.subheader("Detection/Tracking Trend")
        trend_df = stats_df.set_index("frame_idx")[["num_detections", "num_tracks"]]
        st.line_chart(trend_df)

    if result.annotated_video_path and Path(result.annotated_video_path).exists():
        st.subheader("Annotated Preview")
        st.video(result.annotated_video_path)
        st.caption(f"Annotated video saved to {result.annotated_video_path}")

    st.subheader("Event Log")
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download events CSV", csv_bytes, file_name="events.csv")
    else:
        st.info("No events produced by the current settings.")


if __name__ == "__main__":
    main()
