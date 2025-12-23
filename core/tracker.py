import os
import json
import tempfile
import numpy as np
import cv2
import torch
from collections import deque
from paddlex import create_model
from ultralytics.trackers.bot_sort import BOTSORT
from types import SimpleNamespace
from ultralytics.engine.results import Boxes
from scipy.fft import fft
from . import features as feat_lib
from . import kalman as kf_lib
from utils import filters


class VideoTracker:
    def __init__(self, model_name, model_dir, conf_threshold=0.01):
        self.model = create_model(model_name=model_name, model_dir=str(model_dir))
        self.conf_threshold = conf_threshold
        self.num_interpolate_points = 100
        self.cross_frame_window = 5

        tracker_config = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.2, 'track_low_thresh': 0.05,
            'new_track_thresh': 0.5, 'track_buffer': 240,
            'match_thresh': 0.99, 'gmc_method': 'none',
            'proximity_thresh': 0.9, 'appearance_thresh': 0.3,
            'with_reid': True, 'model': 'yolo11n-cls.pt', 'fuse_score': False
        }
        self.tracker = BOTSORT(args=SimpleNamespace(**tracker_config), frame_rate=30)

        self.track_features = {}
        self.track_pharynxes = {}
        self.track_tails = {}
        self.track_centers = {}
        self.track_centerlines = {}
        self.track_history = []
        self.track_directions = {}
        self.track_kalman = {}
        self.track_kalman_centerline = {}
        self.track_raw_features = {}
        self.motion_state_tracker = {}
        self.track_length_history = {}

    def detect_frame(self, frame):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
            cv2.imwrite(tmp_img.name, frame)
            tmp_img_path = tmp_img.name

        try:
            outputs = list(self.model.predict(tmp_img_path, batch_size=1))
            if not outputs:
                return np.empty((0, 6)), []

            result = outputs[0]
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp_json:
                result.save_to_json(tmp_json.name)
                tmp_json_path = tmp_json.name

            with open(tmp_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            boxes_data = data.get('boxes', [])
            masks_data = data.get('masks', [])
            detections = []
            masks_list = []
            h, w = frame.shape[:2]

            for i, item in enumerate(boxes_data):
                bbox = item.get('coordinate', item.get('bbox', []))
                score = item.get('score', 0.0)
                label = item.get('cls_id', item.get('category_id', 0))

                if score < self.conf_threshold: continue

                if len(bbox) == 4:
                    detections.append([*bbox, score, label])

                    full_mask = None
                    if i < len(masks_data):
                        mask_raw = masks_data[i]
                        mask_arr = np.array(mask_raw, dtype=np.uint8)
                        if mask_arr.ndim == 2:
                            x1, y1, x2, y2 = map(int, bbox)
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                            bbox_w, bbox_h = x2 - x1, y2 - y1
                            if bbox_w > 0 and bbox_h > 0:
                                full_mask = np.zeros((h, w), dtype=np.uint8)
                                resized = cv2.resize((mask_arr * 255).astype(np.uint8), (bbox_w, bbox_h),
                                                     interpolation=cv2.INTER_NEAREST)
                                full_mask[y1:y2, x1:x2] = resized
                    masks_list.append(full_mask)

            os.unlink(tmp_json_path)
            return np.array(detections) if detections else np.empty((0, 6)), masks_list
        finally:
            if os.path.exists(tmp_img_path):
                os.unlink(tmp_img_path)

    def process_features(self, track_id, mask, centerline_points, frame_rate):
        raw = {}

        raw['area'] = int(np.sum(mask > 0))
        length = 0
        if len(centerline_points) > 1:
            for i in range(len(centerline_points) - 1):
                length += np.linalg.norm(np.array(centerline_points[i + 1]) - np.array(centerline_points[i]))
        raw['length'] = float(length)
        raw['width'] = float(raw['area'] / length) if length > 0 else 0.0

        if len(centerline_points) > 1:
            s_dist = np.linalg.norm(np.array(centerline_points[-1]) - np.array(centerline_points[0]))
            raw['overall_curvature'] = length / s_dist if s_dist > 0 else 1.0
        else:
            raw['overall_curvature'] = 1.0

        seg_curvs = []
        if len(centerline_points) >= 10:
            pts_per_seg = len(centerline_points) // 10
            for i in range(10):
                s_idx = i * pts_per_seg
                e_idx = (i + 1) * pts_per_seg if i < 9 else len(centerline_points)
                seg_curvs.append(feat_lib.calculate_curvature(centerline_points[s_idx:e_idx]))
        else:
            seg_curvs = [0.0] * 10
        raw['segment_curvatures'] = seg_curvs

        raw['asymmetry'] = feat_lib.calculate_asymmetry(mask, centerline_points)

        if track_id not in self.track_kalman:
            self.track_kalman[track_id] = {'pharynx': kf_lib.init_kalman(), 'tail': kf_lib.init_kalman(),
                                           'center': kf_lib.init_kalman()}

        if len(centerline_points) > 0:
            pharynx_pos = np.array(centerline_points[max(0, int(len(centerline_points) * 0.1))])
            tail_pos = np.array(centerline_points[-1])
            center_pos = np.array(centerline_points[len(centerline_points) // 2])

            for key, pos in zip(['pharynx', 'tail', 'center'], [pharynx_pos, tail_pos, center_pos]):
                kf = self.track_kalman[track_id][key]
                kf.predict()
                kf.update(pos.reshape(2, 1))
                raw[f'{key}_position'] = kf.x[:2].flatten().tolist()

            self.track_pharynxes[track_id] = np.array(raw['pharynx_position'])
            if track_id not in self.track_centerlines: self.track_centerlines[track_id] = deque(maxlen=30)
            self.track_centerlines[track_id].append(centerline_points)

        dt = 1.0 / frame_rate
        speed_fwd, speed_bwd = 0.0, 0.0

        if len(self.track_centerlines[track_id]) >= 2:
            prev_line = self.track_centerlines[track_id][-2]
            curr_line = centerline_points

            p_idx = max(0, int(len(prev_line) * 0.1))
            disp = np.array(curr_line[p_idx]) - np.array(prev_line[p_idx])

            body_vec = np.array(prev_line[-1]) - np.array(prev_line[p_idx])
            body_dir = body_vec / (np.linalg.norm(body_vec) + 1e-6)

            proj_speed = np.dot(disp, body_dir) / dt

            if proj_speed > 0:
                speed_fwd = proj_speed
            elif proj_speed < 0:
                speed_bwd = proj_speed

        raw['speed_forward'] = float(speed_fwd)
        raw['speed_backward'] = float(speed_bwd)
        raw['speed'] = float(speed_fwd if speed_fwd > 0 else speed_bwd)

        if track_id not in self.motion_state_tracker:
            self.motion_state_tracker[track_id] = {'prev': None, 'dur': 0}

        curr_motion = 'fwd' if speed_fwd > 1.0 else ('bwd' if abs(speed_bwd) > 1.0 else 'stat')
        is_reversal = False
        if self.motion_state_tracker[track_id]['prev'] == 'fwd' and curr_motion == 'bwd':
            if self.motion_state_tracker[track_id]['dur'] >= 2:
                is_reversal = True

        if curr_motion != self.motion_state_tracker[track_id]['prev']:
            self.motion_state_tracker[track_id]['prev'] = curr_motion
            self.motion_state_tracker[track_id]['dur'] = 1
        else:
            self.motion_state_tracker[track_id]['dur'] += 1
        raw['reversal'] = int(is_reversal)

        raw['omega_turn'] = 0
        if len(centerline_points) > 5:
            p_pos = np.array(raw['pharynx_position'])
            t_pos = np.array(raw['tail_position'])
            dist = np.linalg.norm(p_pos - t_pos)
            if dist / (length + 1e-6) < 0.5:
                raw['omega_turn'] = 1

        return raw

    def update(self, frame, frame_id, frame_rate=30):
        detections, masks = self.detect_frame(frame)

        active_tracks = []
        if len(detections) > 0:
            data = torch.from_numpy(detections).float()
            boxes = Boxes(data, orig_shape=frame.shape[:2])
            tracks = self.tracker.update(boxes, frame)  # BoTSORT update

            for i, track in enumerate(tracks):
                if len(track) < 5: continue
                track_id = int(track[4])

                track_box = track[:4]
                best_iou, best_mask = 0, None
                for det_idx, det in enumerate(detections):
                    det_box = det[:4]
                    xA = max(track_box[0], det_box[0])
                    yA = max(track_box[1], det_box[1])
                    xB = min(track_box[2], det_box[2])
                    yB = min(track_box[3], det_box[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter > 0:
                        iou = inter / ((track_box[2] - track_box[0]) * (track_box[3] - track_box[1]) + (
                                    det_box[2] - det_box[0]) * (det_box[3] - det_box[1]) - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_mask = masks[det_idx]

                features = {}
                centerline = []
                if best_mask is not None and best_iou > 0.5:
                    skel, centerline = feat_lib.extract_skeleton(best_mask)

                    if len(centerline) > 1:
                        if track_id in self.track_pharynxes:
                            last_p = self.track_pharynxes[track_id]
                            d1 = np.linalg.norm(np.array(centerline[0]) - last_p)
                            d2 = np.linalg.norm(np.array(centerline[-1]) - last_p)
                            if d2 < d1: centerline = centerline[::-1]
                        else:
                            pass

                        features = self.process_features(track_id, best_mask, centerline, frame_rate)

                if features:
                    self.track_history.append({
                        'frame_id': frame_id, 'track_id': track_id, 'features': features
                    })
                    if track_id not in self.track_features: self.track_features[track_id] = deque(maxlen=100)
                    self.track_features[track_id].append(features)

                active_tracks.append({
                    'id': track_id, 'box': track_box,
                    'centerline': centerline, 'features': features
                })

        return active_tracks