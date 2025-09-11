from __future__ import annotations
import os
from typing import List, Dict, Any
from .events import write_events_csv

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_events(out_dir: str, events: List[Dict[str,Any]], stem: str = "events"):
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f"{stem}.csv")
    return write_events_csv(events, out_csv)
