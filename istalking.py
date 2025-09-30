#!/usr/bin/env python3
"""
Inner Lip / Speaking Activity Tracker

Tracks lip motion over time by measuring the rate of change
between upper- and lower-lip means. The track with the
highest activity is marked as "speaking".
"""

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Missing: mediapipe. Install: pip install mediapipe")

# ---------------------------- Landmark sets ----------------------------
INNER_LIP_CONTOUR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

UPPER_INNER = [13, 82, 81, 80, 191]
LOWER_INNER = [14, 87, 178, 317, 402, 318, 324]

# ------------------------------ Utils ---------------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter + 1e-9
    return inter / union

def bbox_from_landmarks(lm_norm: np.ndarray, w: int, h: int, pad: float = 0.15):
    xs = lm_norm[:, 0] * w
    ys = lm_norm[:, 1] * h
    x1, y1 = float(np.min(xs)), float(np.min(ys))
    x2, y2 = float(np.max(xs)), float(np.max(ys))
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    x1, y1 = max(0.0, x1 - pw), max(0.0, y1 - ph)
    x2, y2 = min(float(w - 1), x2 + pw), min(float(h - 1), y2 + ph)
    return (x1, y1, x2, y2)

def to_px(landmarks_norm: np.ndarray, idxs: List[int], w: int, h: int) -> np.ndarray:
    pts = []
    for i in idxs:
        if 0 <= i < landmarks_norm.shape[0]:
            pts.append([landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)

# ----------------------------- Tracker --------------------------------
@dataclass
class InnerLipTrack:
    id: int
    box: Tuple[float, float, float, float]
    last_seen_frame: int
    frames_seen: int = 0
    ready: bool = False

    lip_openings: deque = field(default_factory=lambda: deque(maxlen=90))
    speaking_conf: float = 0.0

class InnerLipTracker:
    def __init__(self, iou_thresh: float = 0.3, min_frames: int = 3):
        self.iou_thresh = iou_thresh
        self.min_frames = min_frames
        self.tracks: Dict[int, InnerLipTrack] = {}
        self.next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float]], frame_idx: int):
        unmatched = set(range(len(detections)))
        assigned: Dict[int, Tuple[float, float, float, float]] = {}

        for tid, tr in list(self.tracks.items()):
            best_j, best_i = None, 0.0
            for j in list(unmatched):
                i = iou(tr.box, detections[j])
                if i > best_i:
                    best_i, best_j = i, j
            if best_j is not None and best_i >= self.iou_thresh:
                tr.box = detections[best_j]
                tr.last_seen_frame = frame_idx
                tr.frames_seen += 1
                tr.ready = tr.ready or (tr.frames_seen >= self.min_frames)
                assigned[tid] = tr.box
                unmatched.remove(best_j)

        for j in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = InnerLipTrack(
                id=tid, box=detections[j], last_seen_frame=frame_idx,
                frames_seen=1, ready=(self.min_frames <= 1)
            )
            assigned[tid] = detections[j]

        stale_cutoff = frame_idx - 90
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].last_seen_frame < stale_cutoff:
                del self.tracks[tid]

        return assigned

# --------------------------- Visualization ----------------------------
def draw_all_landmarks(frame, lm_norm, w, h, color=(0, 150, 255)):
    for i in range(lm_norm.shape[0]):
        x, y = int(lm_norm[i, 0] * w), int(lm_norm[i, 1] * h)
        cv2.circle(frame, (x, y), 1, color, -1)

def draw_overlay(frame, width, height, people, matched_landmarks):
    cv2.putText(frame, f"Faces: {len(people)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, person in enumerate(people):
        x1, y1, x2, y2 = map(int, person["box"])
        color = (0, 255, 0) if person["speaking"] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{person['id']} conf:{person['speaking_conf']:.2f}"
        if person["speaking"]:
            label += " SPEAKING"

        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if matched_landmarks and i < len(matched_landmarks):
            draw_all_landmarks(frame, matched_landmarks[i], width, height)

# ------------------------------- Main ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Speaking Detector (lip opening rate)")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", default="inner_lip_movement.json", help="Output JSON path")
    ap.add_argument("--max-faces", type=int, default=10)
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--viz-inner-lip", action="store_true")
    ap.add_argument("--buffer-frames", type=int, default=90,
                    help="Buffer length in frames")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.video} ({width}x{height}, {fps:.1f} FPS)")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=args.max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2

    )

    tracker = InnerLipTracker(iou_thresh=0.25, min_frames=2)
    timeline = []
    frame_idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            detections, faces_lm = [], []
            if res.multi_face_landmarks:
                for flm in res.multi_face_landmarks:
                    lm = np.array([[p.x, p.y] for p in flm.landmark], dtype=np.float32)
                    faces_lm.append(lm)
                    detections.append(bbox_from_landmarks(lm, width, height))

            assigned = tracker.update(detections, frame_idx)
            people, matched_lm = [], []

            for tid, box in assigned.items():
                tr = tracker.tracks.get(tid)
                if not tr: continue

                best_idx, best_iou = None, 0.0
                for i, lm in enumerate(faces_lm):
                    b = bbox_from_landmarks(lm, width, height)
                    iou_val = iou(box, b)
                    if iou_val > best_iou:
                        best_idx, best_iou = i, iou_val
                if best_idx is None:
                    continue

                lm = faces_lm[best_idx]
                matched_lm.append(lm)

                # --- compute lip opening ---
                upper_pts = to_px(lm, UPPER_INNER, width, height)
                lower_pts = to_px(lm, LOWER_INNER, width, height)
                if len(upper_pts) > 0 and len(lower_pts) > 0:
                    upper_mean = np.mean(upper_pts, axis=0)
                    lower_mean = np.mean(lower_pts, axis=0)
                    dist = np.linalg.norm(upper_mean - lower_mean)
                    box_h = tr.box[3] - tr.box[1]   # bounding box height in pixels
                    if box_h > 0:
                        dist /= box_h
                    tr.lip_openings.append(dist)


                if len(tr.lip_openings) > 2:
                    diffs = np.abs(np.diff(tr.lip_openings))
                    tr.speaking_conf = float(np.mean(diffs))
                else:
                    tr.speaking_conf = 0.0

                people.append({
                    "id": tid,
                    "box": tuple(map(float, box)),
                    "speaking_conf": tr.speaking_conf,
                    "speaking": False
                })

            if people:
                best = max(people, key=lambda p: p["speaking_conf"])
                if best["speaking_conf"] > 0:
                    best["speaking"] = True

            t_sec = frame_idx / fps
            timeline.append({"time": t_sec, "people": people})

            if args.show:
                vis = frame.copy()
                if args.viz_inner_lip:
                    draw_overlay(vis, width, height, people, matched_lm)
                else:
                    draw_overlay(vis, width, height, people, None)
                cv2.imshow("Speaking Detector", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

    if args.out == "inner_lip_movement.json":
        outfile = args.video + ".json"
    else:
        outfile = args.out
    with open(args.out, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
