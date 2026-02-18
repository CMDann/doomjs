# YOLO Studio

A professional, full-featured desktop application for training, managing, and remotely deploying
YOLO computer vision models. Built with PyQt6 and the Ultralytics library.

---

## Features

| Tab | Description |
|-----|-------------|
| **Train** | Configure and launch YOLO training runs with live loss/mAP charts |
| **Datasets** | Build, import, browse, and export YOLO-format datasets. Browse saved model weights. |
| **Discover** | Search Roboflow Universe and HuggingFace Hub; download datasets and model weights |
| **Remote Devices** | Manage Jetson/Xavier/Raspberry Pi edge devices; deploy models and run remote inference |

---

## Architecture

```
yolo_studio/
├── main.py                      # Entry point — launches QApplication
├── requirements.txt
├── pyproject.toml
├── assets/
│   └── icons/                   # SVG/PNG toolbar icons
├── core/
│   ├── database.py              # SQLAlchemy ORM models + session helpers
│   ├── trainer.py               # YOLOTrainer QThread — non-blocking training
│   ├── remote_manager.py        # WebSocket client for edge device communication
│   └── dataset_manager.py       # Dataset CRUD and import/export helpers
├── ui/
│   ├── main_window.py           # QMainWindow: tab container + status bar + log dock
│   ├── theme.py                 # Dark QSS stylesheet + apply_theme()
│   ├── tabs/
│   │   ├── train_tab.py
│   │   ├── dataset_tab.py
│   │   ├── discover_tab.py
│   │   └── remote_tab.py
│   └── widgets/
│       ├── log_panel.py         # Colour-coded scrollable log widget
│       ├── metric_chart.py      # Live pyqtgraph training curves
│       ├── file_drop_zone.py    # Drag-and-drop upload zone
│       └── device_card.py       # Edge device status card
└── edge/
    ├── jetson_agent.py          # WebSocket server to run on edge devices
    └── agent_config.yaml        # Edge agent configuration template

Data flow:

  ┌─────────────┐      SQLAlchemy      ┌──────────────┐
  │   PyQt6 UI  │ ◄──────────────────► │  SQLite DB   │
  └──────┬──────┘                      └──────────────┘
         │ QThread signals
  ┌──────▼──────┐     ultralytics      ┌──────────────┐
  │  trainer.py │ ◄──────────────────► │  YOLO model  │
  └─────────────┘                      └──────────────┘
         │ WebSocket (JSON)
  ┌──────▼──────────────┐
  │  jetson_agent.py    │  (runs on Jetson / Xavier / Pi)
  └─────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- CUDA toolkit (optional, for GPU training) — follow the [PyTorch install guide](https://pytorch.org/get-started/locally/)
- On Linux/macOS: `libgl1-mesa-glx` (for OpenCV headless support)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd yolo_studio

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch YOLO Studio
python main.py
```

### First-run Configuration

On first launch YOLO Studio creates a `config.json` in the working directory.
Populate it with your API keys:

```json
{
  "roboflow_api_key": "YOUR_ROBOFLOW_KEY",
  "huggingface_token": "YOUR_HF_TOKEN"
}
```

These values can also be entered directly in the **Discover** tab UI and will be saved automatically.

---

## Usage Guide

### Train Tab

1. Select a model architecture (e.g. `yolov8n`) from the dropdown.
2. Choose a dataset from the library (or drag image folders into the drop zones).
3. Adjust hyperparameters — sensible defaults are pre-filled.
4. Click **Start Training**. Live loss/mAP curves appear on the right panel.
5. When satisfied, click **Save This Run** to persist weights to `saved_models/`.

### Dataset Tab

- **Library** (left): Browse all datasets registered in the local SQLite database.
  Right-click a row for Edit / Delete / Export / View options.
- **Builder** (right): Add images, manage class labels, set train/val/test splits,
  generate `data.yaml`, then save to the library.
- **Saved Models** sub-tab: Inspect, export, or push model weights to a connected device.

### Discover Tab

- Toggle between **Roboflow Universe** and **HuggingFace Hub** using the segmented button bar.
- Enter your API key/token once; it is saved to `config.json`.
- Click **Download Dataset** or **Download Model** on any result card.
  Downloaded assets are automatically registered in the local database.

### Remote Devices Tab

- Click **Add Device** and enter the edge device's IP, port, and auth token.
- **Ping All** verifies connectivity and updates each device's status badge.
- In the **Test Runner** panel: select a device, a saved model, and a test dataset,
  then click **Deploy & Test**. Results stream back and are stored in the database.

---

## Edge Agent Setup (Jetson / Xavier / Raspberry Pi)

```bash
# On the edge device:
pip install ultralytics websockets PyYAML

# Copy the agent files
scp edge/jetson_agent.py edge/agent_config.yaml user@device-ip:~/yolo_agent/

# Edit the config
nano ~/yolo_agent/agent_config.yaml   # set auth_token, paths, port

# Run the agent
cd ~/yolo_agent
python jetson_agent.py

# (Optional) Install as a systemd service — see the comment block at the top of jetson_agent.py
```

Make sure the port specified in `agent_config.yaml` matches the one you enter
in YOLO Studio's **Add Device** dialog.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: PyQt6` | Run `pip install -r requirements.txt` in your virtual environment |
| Black / blank window on Linux | Set `QT_QPA_PLATFORM=xcb` or install `libxcb-xinerama0` |
| Training never starts | Check that `ultralytics` can import successfully: `python -c "from ultralytics import YOLO"` |
| Device shows Offline after Ping | Verify firewall rules allow the WebSocket port; confirm the agent is running |
| Roboflow download fails | Confirm your API key in the Discover tab; check Roboflow project visibility |
| `CUDA out of memory` | Reduce batch size or image size in the Train tab hyperparameter form |

---

## Contributing

Pull requests are welcome. Please ensure:
- All new classes and public methods have Google-style docstrings.
- Type hints are present on all function signatures.
- Long-running operations run in `QThread` subclasses — no blocking the main thread.
- `config.json`, `*.pt`, `datasets/`, and `runs/` are listed in `.gitignore` and never committed.

---

## License

MIT License — see `LICENSE` for details.
