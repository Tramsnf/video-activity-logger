from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    app_path = src_path / "vap" / "web" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

    try:
        from streamlit.web import bootstrap
    except ImportError as exc:
        raise RuntimeError("Streamlit is not installed. Run `pip install -r requirements.txt`." ) from exc

    # Launch the Streamlit app programmatically so users can simply run
    # `python scripts/run_studio.py`.
    bootstrap.run(str(app_path), False, [], {})


if __name__ == "__main__":
    main()
