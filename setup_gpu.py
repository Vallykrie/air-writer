"""
Configuration file untuk Air Writer
Ubah nilai di sini untuk menyesuaikan sensitivity dan behavior
"""

# ==================== PERFORMANCE SETTINGS ====================

# GPU Acceleration (NVIDIA CUDA)
USE_GPU = True  # Set False jika tidak punya NVIDIA GPU
GPU_DEVICE = 0  # GPU device index (0 untuk primary GPU)

# MediaPipe Performance Mode
# 0 = Accuracy (slow), 1 = Fast (recommended)
MODEL_COMPLEXITY = 0  # 0=lite, 1=full (gunakan 0 untuk speed)

# Frame Processing
SKIP_FRAMES = 0  # Skip N frames untuk speed boost (0=no skip, 1=skip every other)
REDUCE_RESOLUTION = False  # Reduce camera resolution untuk speed
PROCESS_WIDTH = 640  # Processing width jika REDUCE_RESOLUTION=True
PROCESS_HEIGHT = 480  # Processing height jika REDUCE_RESOLUTION=True

# ==================== GESTURE SENSITIVITY ====================

# PINCH DETECTION
# Semakin KECIL nilai = semakin SENSITIF
# Recommended range: 20-50 pixels
PINCH_THRESHOLD = 30  # Default: 30, Lebih sensitif: 20, Kurang sensitif: 40

# PEACE SIGN (✌️) DETECTION
# Minimum jarak antara jari untuk dianggap terpisah
PEACE_FINGER_SEPARATION = 30  # Default: 30

# POINTING (☝️) DETECTION
# Berapa persen jari harus lebih tinggi dari base untuk dianggap extended
FINGER_EXTENSION_RATIO = 1.2  # Default: 1.2 (20% lebih tinggi)

# ==================== MEDIAPIPE SETTINGS ====================

# Detection confidence (0.0 - 1.0)
# Semakin tinggi = lebih strict dalam deteksi tangan
MIN_DETECTION_CONFIDENCE = 0.5  # Turunkan dari 0.7 untuk speed

# Tracking confidence (0.0 - 1.0)
# Semakin tinggi = tracking lebih stable tapi mungkin lebih lambat
MIN_TRACKING_CONFIDENCE = 0.5  # Default: 0.5

# Maximum number of hands to detect
MAX_NUM_HANDS = 1  # Default: 1 (single hand)

# ==================== DRAWING SETTINGS ====================

# Line smoothing buffer size (1-10)
# Semakin besar = garis lebih smooth tapi ada delay
SMOOTHING_BUFFER_SIZE = 3  # Turunkan dari 5 untuk speed

# Drawing colors (BGR format)
COLORS = [
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Green
    (255, 255, 0),   # Yellow
    (255, 255, 255), # White
    (255, 0, 0),     # Blue
    (0, 165, 255),   # Orange
    (0, 0, 255),     # Red
]

COLOR_NAMES = ['Cyan', 'Magenta', 'Green', 'Yellow', 'White', 'Blue', 'Orange', 'Red']

# Brush sizes (pixels)
BRUSH_SIZES = [3, 5, 8, 12, 16]

# Default selections
DEFAULT_COLOR_INDEX = 0
DEFAULT_BRUSH_INDEX = 1

# Canvas overlay transparency (0.0 - 1.0)
CANVAS_ALPHA = 0.8  # Default: 0.8

# ==================== GESTURE COOLDOWN ====================

# Cooldown untuk prevent accidental triggers (seconds)
CLEAR_COOLDOWN = 1.5  # Peace sign cooldown
UNDO_COOLDOWN = 0.5   # Pointing cooldown

# ==================== CAMERA SETTINGS ====================

CAMERA_INDEX = 0  # Default camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Mirror camera feed
MIRROR_CAMERA = True  # True = selfie mode

# ==================== UNDO SETTINGS ====================

# Maximum undo history
MAX_UNDO_HISTORY = 20  # Default: 20 steps

# ==================== UI SETTINGS ====================

# Show FPS counter
SHOW_FPS = True

# Show hand landmarks (disable untuk speed)
SHOW_LANDMARKS = True

# Show gesture indicators
SHOW_GESTURE_HINTS = True

# ==================== DEBUG MODE ====================

DEBUG_MODE = False  # Set True untuk melihat debug info
VERBOSE = False     # Set True untuk console logging yang detail