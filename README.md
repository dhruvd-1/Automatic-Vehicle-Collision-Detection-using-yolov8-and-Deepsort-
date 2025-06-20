# Vehicle Collision Detection System

A real-time vehicle collision detection system using YOLOv8 for object detection and DeepSORT for multi-object tracking. The system monitors live camera feeds and automatically detects vehicle collisions, sending emergency alerts when incidents occur.

## 📁 Project Structure

```
vehicle-collision-detection/
│
├── 📄 pipeline.py              # Main system orchestrator & entry point
├── 📄 detector.py              # YOLOv8 vehicle detection module
├── 📄 tracker.py               # DeepSORT multi-object tracking
├── 📄 collision.py             # Collision detection algorithms
├── 📄 alert.py                 # SMS alert & logging system
├── 📄 config.yaml              # System configuration file
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project documentation
│
├── 📁 utils/                   # Utility modules
│   └── 📄 draw_utils.py       # Visualization & drawing functions
│
├── 📁 models/                  # Model files (auto-created)
│   └── 📄 yolov8n.pt          # YOLOv8 weights (auto-downloaded)
│
├── 📁 logs/                    # Generated log files
│   ├── 📄 collision_log.csv   # Collision event logs
│   └── 📄 system.log          # System operation logs
│
├── 📁 recordings/              # Video recordings (auto-created)
│   ├── 🎥 collision_20250620_143022_vehicles_1_2.mp4
│   ├── 🎥 collision_20250620_143155_vehicles_3_4.mp4
│   └── 🎥 ...
│
├── 📁 snapshots/              # Saved frame images
│   ├── 🖼️ frame_20250620_143022.jpg
│   ├── 🖼️ frame_20250620_143155.jpg
│   └── 🖼️ ...
│
├── 📁 tests/                   # Unit tests (optional)
│   ├── 📄 test_detector.py
│   ├── 📄 test_tracker.py
│   ├── 📄 test_collision.py
│   └── 📄 test_alerts.py
│
└── 📁 docs/                    # Additional documentation
    ├── 📄 installation.md
    ├── 📄 configuration.md
    ├── 📄 troubleshooting.md
    └── 📄 api_reference.md
```

### 📋 File Descriptions

| File/Directory | Purpose | Key Components |
|---------------|---------|----------------|
| `pipeline.py` | Main orchestrator that ties all components together | Camera handling, main loop, UI controls |
| `detector.py` | Vehicle detection using YOLOv8 | Model loading, inference, filtering |
| `tracker.py` | Multi-object tracking with DeepSORT | Track management, trajectory history |
| `collision.py` | Collision detection algorithms | Overlap calculation, motion analysis |
| `alert.py` | Alert system and logging | SMS via Twilio, CSV logging |
| `config.yaml` | Centralized configuration | All system parameters |
| `utils/draw_utils.py` | Visualization functions | Bounding boxes, trails, alerts |
| `models/` | AI model storage | YOLOv8 weights (auto-downloaded) |
| `logs/` | System logs and collision records | CSV files, system logs |
| `recordings/` | Collision video recordings | Auto-generated MP4 files |
| `snapshots/` | Manual frame captures | JPEG images |

### 🔧 Auto-Generated Directories

The following directories are created automatically when the system runs:

- `models/` - Downloads YOLOv8 model weights on first run
- `logs/` - Creates collision logs and system logs
- `recordings/` - Stores collision incident videos
- `snapshots/` - Saves manually captured frames (press 's')

## Features

🚗 **Real-time Vehicle Detection**: Uses YOLOv8 to detect cars, trucks, buses, and motorcycles
🎯 **Multi-Object Tracking**: DeepSORT tracking maintains vehicle identities across frames
⚡ **Collision Detection**: Heuristic-based collision detection using bounding box overlap and motion analysis
📱 **Emergency Alerts**: Automatic SMS notifications via Twilio when collisions are detected
📊 **Data Logging**: CSV logging of all collision events with timestamps and details
🎥 **Video Recording**: Automatic video capture of collision incidents
📈 **Real-time Visualization**: Live annotated video feed with bounding boxes and collision alerts

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV-compatible camera or RTSP stream
- Twilio account (for SMS alerts)

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure the system**:
   - Copy `config.yaml` and edit the configuration
   - Set up camera source (webcam or RTSP URL)
   - Configure Twilio credentials for SMS alerts
   - Adjust detection and collision thresholds

4. **Run the system**:
```bash
python pipeline.py
```
