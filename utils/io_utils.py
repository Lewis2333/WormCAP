import pandas as pd
import json
import numpy as np
from pathlib import Path


def export_data(track_history, track_features_summary, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in track_history:
        row = {
            'frame_id': entry['frame_id'],
            'track_id': entry['track_id']
        }
        feats = entry['features']
        for k, v in feats.items():
            if k == 'segment_curvatures':
                for i, val in enumerate(v):
                    row[f'curvature_seg_{i + 1}'] = val
            elif isinstance(v, (list, tuple)):
                pass
            else:
                row[k] = v
        records.append(row)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_path / 'per_frame_features.csv', index=False)
        print(f"✅ CSV Saved: {out_path / 'per_frame_features.csv'}")

    summary = {}
    for tid, feats_deque in track_features_summary.items():
        if not feats_deque: continue
        all_feats = list(feats_deque)
        avg_data = {}
        keys = all_feats[0].keys()
        for k in keys:
            if isinstance(all_feats[0][k], (int, float)):
                vals = [f[k] for f in all_feats if k in f]
                avg_data[k] = float(np.mean(vals))
        summary[tid] = avg_data

    with open(out_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ JSON Saved: {out_path / 'summary.json'}")