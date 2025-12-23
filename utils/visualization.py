import cv2
import numpy as np


def draw_tracks(frame, tracks, frame_id):
    vis_frame = frame.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    cv2.putText(vis_frame, f"Frame: {frame_id} | Count: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for trk in tracks:
        tid = trk['id']
        box = list(map(int, trk['box']))
        color = colors[tid % len(colors)]

        cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(vis_frame, f"ID:{tid}", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = trk['centerline']
        if len(pts) > 1:
            pts_arr = np.array(pts, np.int32)
            cv2.polylines(vis_frame, [pts_arr], False, (0, 255, 0), 2)
            cv2.circle(vis_frame, tuple(map(int, pts[0])), 4, (0, 0, 255), -1)
            cv2.circle(vis_frame, tuple(map(int, pts[-1])), 4, (255, 0, 0), -1)
        feat = trk['features']
        if feat:
            info = f"Speed:{feat.get('speed', 0):.1f} Rev:{feat.get('reversal', 0)}"
            cv2.putText(vis_frame, info, (box[0], box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis_frame