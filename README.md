# clungevoters
A system to count votes using openCV
## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run with default webcam (index `0`):

```bash
python app.py
```

Optional parameters:

```bash
python app.py --camera-index 0 --width 960 --height 540 --detection-stride 2
```

### CLI Options

- `--camera-index` (int): Camera index for OpenCV capture
- `--width` (int): Requested capture width
- `--height` (int): Requested capture height
- `--detection-stride` (int): Run person detector every N frames (higher = faster but less responsive)

## Controls During Runtime

- **F**: Freeze/unfreeze displayed vote count
  - When freezing, the current live count is captured as a snapshot
- **R**: Reset (returns to live counting mode and clears snapshot)
- **Q** or **Esc**: Quit application

## How Raised Hand Detection Works

1. Person detector finds candidate person bounding boxes in each frame.
2. For each person ROI, MediaPipe Pose estimates body landmarks.
3. A hand is considered raised if a wrist is above its shoulder (with margin) and approximately above face level.
4. A person is counted as a voter if either left or right hand is raised.

## Tips for Better Accuracy

- Use good lighting and keep people mostly visible in frame.
- Keep some space between people to reduce overlap.
- Position camera so upper body and arms are clearly visible.
- Reduce background clutter when possible.

## Known Limitations

- HOG person detection can miss people at extreme angles or distances.
- Very crowded scenes may reduce reliability.
- Occlusions (arms hidden behind others/objects) can cause missed detections.
- For higher accuracy in large rooms, consider replacing HOG with a stronger detector (e.g., YOLO-based person detection).

## Troubleshooting

- **Camera does not open**: try another index, e.g. `--camera-index 1`.
- **Low FPS**: increase `--detection-stride` (e.g. `3` or `4`) or lower resolution.
- **No detections**: improve lighting and ensure full upper-body visibility.
