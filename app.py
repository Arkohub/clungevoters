import argparse
import importlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as mp_import_error:  # pragma: no cover - import failure path
    mp = None
    _MP_IMPORT_ERROR = mp_import_error
else:
    _MP_IMPORT_ERROR = None


def resolve_mediapipe_pose_module():
    """Return MediaPipe pose module with compatibility fallbacks."""
    if mp is None:
        raise ImportError(
            "MediaPipe could not be imported. Install dependencies with `pip install -r requirements.txt`. "
            f"Original error: {_MP_IMPORT_ERROR}"
        )

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose

    # Compatibility fallback where `solutions` exists as a submodule but not a top-level attribute.
    try:
        mp_solutions = importlib.import_module("mediapipe.solutions")
        if hasattr(mp_solutions, "pose"):
            return mp_solutions.pose
    except Exception:
        pass

    # Older package layouts may expose pose via mediapipe.python.solutions.pose.
    try:
        return importlib.import_module("mediapipe.python.solutions.pose")
    except Exception as fallback_error:
        mp_location = getattr(mp, "__file__", "unknown location")
        raise ImportError(
            "MediaPipe is installed but Pose APIs are unavailable (missing `mediapipe.solutions.pose`). "
            "This app currently relies on the classic Solutions Pose API; newer task-only builds are not compatible. "
            "Use a stable 0.10.x build such as: `pip uninstall -y mediapipe && pip install mediapipe==0.10.9`, "
            "and ensure your project has no file/folder named `mediapipe`. "
            f"Loaded mediapipe from: {mp_location}."
        ) from fallback_error


@dataclass
class PersonVoteResult:
    bbox: Tuple[int, int, int, int]
    has_raised_hand: bool
    raised_wrists: List[Tuple[int, int]]


class RaisedHandVoteCounter:
    """Detect people and estimate raised hands for vote counting."""

    def __init__(
        self,
        detection_stride: int = 2,
        hand_raise_px_offset: float = 0.05,
        min_visibility: float = 0.45,
    ) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.mp_pose = resolve_mediapipe_pose_module()
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as pose_init_error:
            raise RuntimeError(
                "MediaPipe Pose failed to initialize. Verify a compatible install with "
                "`pip install mediapipe==0.10.9` (or another 0.10.x release)."
            ) from pose_init_error

        self.detection_stride = max(1, detection_stride)
        self.hand_raise_px_offset = hand_raise_px_offset
        self.min_visibility = min_visibility
        self._frame_idx = 0
        self._cached_boxes: List[Tuple[int, int, int, int]] = []

    def close(self) -> None:
        self.pose.close()

    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Run person detector every N frames to improve FPS.
        if self._frame_idx % self.detection_stride == 0 or not self._cached_boxes:
            boxes, weights = self.hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )

            filtered: List[Tuple[int, int, int, int]] = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight < 0.35:
                    continue
                filtered.append((x, y, w, h))

            self._cached_boxes = self._non_max_suppression(filtered, overlap_thresh=0.45)

        self._frame_idx += 1
        return self._cached_boxes

    @staticmethod
    def _non_max_suppression(
        boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []

        arr = np.array(boxes, dtype=np.float32)
        x1 = arr[:, 0]
        y1 = arr[:, 1]
        x2 = arr[:, 0] + arr[:, 2]
        y2 = arr[:, 1] + arr[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        pick = []
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            idxs = idxs[:-1]

            if len(idxs) == 0:
                break

            xx1 = np.maximum(x1[last], x1[idxs])
            yy1 = np.maximum(y1[last], y1[idxs])
            xx2 = np.minimum(x2[last], x2[idxs])
            yy2 = np.minimum(y2[last], y2[idxs])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs]
            idxs = idxs[overlap <= overlap_thresh]

        return [boxes[i] for i in pick]

    def _landmark_point(
        self,
        landmarks,
        landmark_enum,
        crop_w: int,
        crop_h: int,
    ) -> Optional[Tuple[int, int, float]]:
        lm = landmarks[landmark_enum.value]
        if lm.visibility < self.min_visibility:
            return None
        return int(lm.x * crop_w), int(lm.y * crop_h), float(lm.visibility)

    def detect_raised_hand_for_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> PersonVoteResult:
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        margin_x = int(0.08 * w)
        margin_y = int(0.08 * h)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_w, x + w + margin_x)
        y2 = min(frame_h, y + h + margin_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PersonVoteResult(bbox=bbox, has_raised_hand=False, raised_wrists=[])

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb_crop)

        if not pose_result.pose_landmarks:
            return PersonVoteResult(bbox=bbox, has_raised_hand=False, raised_wrists=[])

        lms = pose_result.pose_landmarks.landmark
        crop_h, crop_w = crop.shape[:2]
        px_offset = int(self.hand_raise_px_offset * crop_h)

        def get_point(lm_enum):
            return self._landmark_point(lms, lm_enum, crop_w, crop_h)

        nose = get_point(self.mp_pose.PoseLandmark.NOSE)

        left_shoulder = get_point(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_point(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_wrist = get_point(self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_point(self.mp_pose.PoseLandmark.RIGHT_WRIST)

        raised_wrists_global: List[Tuple[int, int]] = []

        def is_raised(wrist_pt, shoulder_pt) -> bool:
            if wrist_pt is None or shoulder_pt is None:
                return False
            wrist_y = wrist_pt[1]
            shoulder_y = shoulder_pt[1]
            above_shoulder = wrist_y < (shoulder_y - px_offset)
            above_face = nose is None or wrist_y < (nose[1] + px_offset)
            return above_shoulder and above_face

        if is_raised(left_wrist, left_shoulder):
            raised_wrists_global.append((x1 + left_wrist[0], y1 + left_wrist[1]))

        if is_raised(right_wrist, right_shoulder):
            raised_wrists_global.append((x1 + right_wrist[0], y1 + right_wrist[1]))

        return PersonVoteResult(
            bbox=bbox,
            has_raised_hand=len(raised_wrists_global) > 0,
            raised_wrists=raised_wrists_global,
        )


def draw_ui(
    frame: np.ndarray,
    current_votes: int,
    live_votes: int,
    freeze_active: bool,
) -> None:
    header_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    freeze_text = "FROZEN" if freeze_active else "LIVE"
    title = f"Raised Hand Vote Counter [{freeze_text}]"
    cv2.putText(frame, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(
        frame,
        f"Displayed vote count: {current_votes}",
        (14, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Live detected votes: {live_votes}",
        (14, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (200, 255, 200),
        2,
    )

    instructions = "Controls: [F] Freeze/Unfreeze  [R] Reset to Live  [Q or ESC] Exit"
    cv2.putText(
        frame,
        instructions,
        (14, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time raised-hand vote counter.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Requested camera width (default: 960).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Requested camera height (default: 540).",
    )
    parser.add_argument(
        "--detection-stride",
        type=int,
        default=2,
        help="Run person detector every N frames (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            "Check webcam permissions or try a different camera index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    vote_counter: Optional[RaisedHandVoteCounter] = None

    try:
        vote_counter = RaisedHandVoteCounter(detection_stride=args.detection_stride)

        freeze_active = False
        frozen_vote_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: could not read frame from camera.")
                break

            people = vote_counter.detect_people(frame)
            person_results: List[PersonVoteResult] = []

            for bbox in people:
                result = vote_counter.detect_raised_hand_for_person(frame, bbox)
                person_results.append(result)

            live_vote_count = sum(1 for r in person_results if r.has_raised_hand)
            displayed_vote_count = frozen_vote_count if freeze_active else live_vote_count

            for res in person_results:
                x, y, w, h = res.bbox
                color = (0, 200, 0) if res.has_raised_hand else (255, 100, 0)
                label = "VOTE" if res.has_raised_hand else "No vote"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                )

                for wrist_xy in res.raised_wrists:
                    cv2.circle(frame, wrist_xy, 8, (0, 255, 255), -1)
                    cv2.circle(frame, wrist_xy, 14, (0, 255, 255), 2)

            draw_ui(
                frame,
                current_votes=displayed_vote_count,
                live_votes=live_vote_count,
                freeze_active=freeze_active,
            )

            cv2.imshow("Raised Hand Vote Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                freeze_active = not freeze_active
                if freeze_active:
                    frozen_vote_count = live_vote_count
            if key == ord("r"):
                freeze_active = False
                frozen_vote_count = 0

    except (ImportError, RuntimeError) as setup_error:
        raise RuntimeError(
            "Failed to initialize MediaPipe for vote counting. "
            "Run `pip install -r requirements.txt` and verify `python -c \"import mediapipe as mp; print(mp.__version__)\"`."
        ) from setup_error
    finally:
        if vote_counter is not None:
            vote_counter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
