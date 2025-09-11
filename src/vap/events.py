from __future__ import annotations
import json
import pandas as pd
from typing import List, Dict, Any

COLUMNS = ["video_id","actor_id","actor_type","activity","start_time_s","end_time_s","duration_s","confidence","source_camera","attributes"]

def write_events_csv(events: List[Dict[str,Any]], out_csv: str):
    if not events:
        # create empty with schema
        df = pd.DataFrame(columns=COLUMNS)
    else:
        # normalize attributes to JSON string for CSV
        evts = []
        for e in events:
            e = dict(e)
            attrs = e.get("attributes", {})
            if isinstance(attrs, str):
                # Assume it's already JSON-like; keep as is
                e["attributes"] = attrs
            else:
                try:
                    e["attributes"] = json.dumps(attrs, ensure_ascii=False)
                except Exception:
                    e["attributes"] = json.dumps({}, ensure_ascii=False)
            evts.append(e)
        df = pd.DataFrame(evts)[COLUMNS]
    df.to_csv(out_csv, index=False)
    return out_csv
