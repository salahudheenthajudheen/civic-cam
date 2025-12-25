# CivicCam 🎥

**AI-Powered Roadside Litter Detection System**

Real-time computer vision system that detects littering incidents using YOLOv8 object detection. Part of a comprehensive surveillance solution for civic authorities.

## Features

- 🎯 **Real-time Object Detection** - YOLOv8-based detection at 15-30 FPS
- 📹 **Multi-source Support** - Webcam, video files, RTSP streams
- 🖥️ **GPU Acceleration** - CUDA, MPS (Apple Silicon), with CPU fallback
- ⚙️ **Configurable** - YAML-based configuration for all settings
- 🔄 **Auto-reconnect** - Graceful handling of camera disconnections
- 📊 **Live Visualization** - Bounding boxes, labels, FPS counter

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download YOLOv8 Model

The model will be downloaded automatically on first run. Alternatively, download manually:

```bash
# The smallest model (recommended for testing)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run Detection

```bash
# Using webcam (default)
python main.py

# Using video file
python main.py --source path/to/video.mp4

# Using RTSP stream
python main.py --source "rtsp://user:pass@ip:port/stream"

# Headless mode (no display)
python main.py --no-display
```

## Project Structure

```
civiccam/
├── config.yaml           # Configuration settings
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── modules/
│   ├── __init__.py       # Package exports
│   ├── detector.py       # YOLOv8 detection
│   ├── video_capture.py  # Video source handling
│   └── utils.py          # Utilities
├── models/               # Model weights
├── data/evidence/        # Saved evidence
└── logs/                 # Log files
```

## Configuration

Edit `config.yaml` to customize:

```yaml
camera:
  source: 0                    # Webcam index, file path, or URL
  fps: 30
  resolution: [1280, 720]

detection:
  model: "yolov8n.pt"          # yolov8n/s/m/l/x.pt
  confidence_threshold: 0.5
  classes_of_interest:
    - person
    - bottle
    - cup
    - car

display:
  show_fps: true
  box_color: [0, 255, 0]       # BGR format (green)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source`, `-s` | Video source | From config |
| `--config`, `-c` | Config file path | `config.yaml` |
| `--model`, `-m` | YOLO model path | From config |
| `--confidence`, `-conf` | Confidence threshold | From config |
| `--no-display` | Headless mode | False |
| `--debug` | Enable debug logging | False |

## Keyboard Controls

When running with display:
- **q** - Quit
- **s** - Save current frame

## Model Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6 MB | Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 52 MB | Medium | Best |

## System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- Any modern CPU

**Recommended:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with CUDA

## Troubleshooting

### Camera not found
```bash
# List available cameras
python -c "from modules.video_capture import list_available_cameras; print(list_available_cameras())"
```

### Low FPS
- Use a smaller model (`yolov8n.pt`)
- Lower resolution in config
- Ensure GPU is being used (check logs)

### Model download fails
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Development Phases

- [x] **Phase 1**: Core Detection Setup ← Current
- [ ] **Phase 2**: Action Recognition
- [ ] **Phase 3**: License Plate Recognition
- [ ] **Phase 4**: Evidence Management
- [ ] **Phase 5**: Review Dashboard
- [ ] **Phase 6**: Analytics

## License

MIT License - See LICENSE file for details.
