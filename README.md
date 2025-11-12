# 🎨 Air Writer - MediaPipe Hand Tracking

**Draw in the air with your hands!** Air Writer is a computer vision application that lets you create digital art using hand gestures, powered by MediaPipe and OpenCV.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.0+-red.svg)

## ✨ Features

- 🤌 **Pinch to Draw** - Use thumb and index finger pinch gesture to draw
- ✌️ **Peace Sign to Clear** - Show peace sign (2 fingers) to clear canvas
- ☝️ **Point to Undo** - Point with index finger to undo last stroke
- 🎨 **8 Colors Available** - Cyan, Magenta, Green, Yellow, White, Blue, Orange, Red
- 🖌️ **5 Brush Sizes** - From 3px to 16px
- 💾 **Save Your Art** - Export drawings with white background
- 📝 **Undo History** - Up to 20 undo steps
- 🎯 **Smooth Drawing** - Built-in smoothing algorithm for natural strokes
- ⚙️ **Highly Configurable** - Easy to adjust sensitivity and behavior

## 📋 Requirements

- Python 3.8 or higher
- Webcam/Camera
- Windows, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/air-writer.git
cd air-writer
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Application

```bash
python main.py
```

### Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| 🤌 **Pinch** | Draw | Touch thumb and index finger together |
| ✌️ **Peace Sign** | Clear Canvas | Extend index and middle fingers |
| ☝️ **Pointing** | Undo | Extend only index finger up |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `C` | Change color |
| `B` | Change brush size |
| `S` | Save canvas to file |
| `R` | Reset/Clear canvas |
| `U` | Undo last stroke |
| `D` | Toggle debug mode |
| `Q` | Quit application |

## ⚙️ Configuration

All settings can be customized in `config.py`:

### Gesture Sensitivity

```python
# Pinch detection (lower = more sensitive)
PINCH_THRESHOLD = 30  # pixels, range: 20-50

# Peace sign detection
PEACE_FINGER_SEPARATION = 30  # pixels

# Pointing detection
FINGER_EXTENSION_RATIO = 1.2  # 20% extension required
```

### MediaPipe Settings

```python
MIN_DETECTION_CONFIDENCE = 0.7  # 0.0 - 1.0
MIN_TRACKING_CONFIDENCE = 0.5   # 0.0 - 1.0
MAX_NUM_HANDS = 1
```

### Drawing Settings

```python
SMOOTHING_BUFFER_SIZE = 2       # 1-10 frames
CANVAS_ALPHA = 0.8              # transparency
MAX_UNDO_HISTORY = 20           # undo steps
```

### Camera Settings

```python
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MIRROR_CAMERA = True  # selfie mode
```

## 📁 Project Structure

```
air-writer/
│
├── main.py                 # Main application entry point
├── gesture_detector.py     # MediaPipe gesture recognition
├── drawing_canvas.py       # Canvas and drawing logic
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 Available Colors

1. Yellow
2. Magenta
3. Green
4. Cyan
5. White
6. Blue
7. Orange
8. Red

## 🖌️ Brush Sizes

- 3px (Extra Small)
- 5px (Small)
- 8px (Medium)
- 12px (Large)
- 16px (Extra Large)

## 🐛 Troubleshooting

### Camera Not Working

```python
# Try different camera index in config.py
CAMERA_INDEX = 0  # Try 1, 2, etc.
```

### Gesture Detection Too Sensitive/Not Sensitive

Adjust thresholds in `config.py`:

```python
# Make pinch MORE sensitive
PINCH_THRESHOLD = 20  # lower value

# Make pinch LESS sensitive
PINCH_THRESHOLD = 40  # higher value
```

### Low FPS/Performance Issues

```python
# Reduce camera resolution in config.py
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Reduce smoothing buffer
SMOOTHING_BUFFER_SIZE = 1
```

### Hand Not Detected

```python
# Lower detection confidence in config.py
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.3
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) by Google for hand tracking
- [OpenCV](https://opencv.org/) for computer vision capabilities
- Inspired by virtual whiteboard and air drawing applications

## 🎥 Demo

[Ongoing Project]

---

**Made with ❤️ and Python**