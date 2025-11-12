import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ==================== KONFIGURASI (MUDAH DIUBAH) ====================

# SENSITIVITY PINCH (pixels) - semakin kecil = semakin sensitif
PINCH_THRESHOLD = 20  # Default: 30, Sensitif: 20, Longgar: 40

# GESTURE COOLDOWN (frames) - jeda antar gesture
GESTURE_COOLDOWN_FRAMES = 15  # Kurangi dari 30 untuk lebih responsive

# LINE SETTINGS
LINE_THICKNESS_DEFAULT = 5
LINE_THICKNESS_MIN = 2
LINE_THICKNESS_MAX = 20

# COLORS (BGR format)
COLORS = [
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
    (255, 255, 255),  # White
    (0, 0, 255),  # Red
]

# SMOOTHING (0.0-1.0) - semakin besar = semakin smooth tapi delay
CURSOR_SMOOTHING = 0.5  # Default: 0.5

# CAMERA SETTINGS
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# =====================================================================

# --- Global Variables ---
line_thickness = LINE_THICKNESS_DEFAULT
color_index = 0
draw_color = COLORS[color_index]

xp, yp = 0, 0  # Previous point for drawing
drawing_state = False
current_cooldown = 0

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower untuk speed
    min_tracking_confidence=0.5,
    model_complexity=0  # 0=lite (FAST!)
)


# --- Helper Functions ---

def get_gesture(hand_landmarks, image_shape):
    """
    Deteksi gesture dari hand landmarks
    Returns: "pinch", "point", "peace", "open", "fist"
    """
    landmarks = hand_landmarks.landmark
    h, w = image_shape[:2]

    # Check finger states
    tip_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    pip_ids = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky PIPs

    fingers_up = []
    for i in range(4):
        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    # GESTURE DETECTION

    # Point ☝️ (Undo)
    if fingers_up == [1, 0, 0, 0]:
        return "point"

    # Peace ✌️ (Clear)
    if fingers_up == [1, 1, 0, 0]:
        return "peace"

    # Open 🖐️ (Cursor mode)
    if fingers_up == [1, 1, 1, 1]:
        return "open"

    # Fist ✊ (Hide cursor)
    if fingers_up == [0, 0, 0, 0]:
        return "fist"

    # PINCH 🤏 (Draw mode)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    # Calculate distance in pixels
    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

    distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

    if distance < PINCH_THRESHOLD:
        return "pinch"

    return "open"  # Default


def get_cursor_position(hand_landmarks, image_shape):
    """Get index finger tip position (cursor)"""
    h, w = image_shape[:2]
    index_tip = hand_landmarks.landmark[8]
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    return (x, y)


# --- Main Program ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
    exit()

drawing_canvas = None
undo_stack = []

# FPS Counter
fps = 0
frame_count = 0
fps_time = time.time()

print("=" * 60)
print("🎨 AIR WRITER - Fast & Lightweight")
print("=" * 60)
print("\n📝 GESTURES:")
print("  🤏 PINCH (thumb+index)  : Draw")
print("  ✌️  PEACE (2 fingers)    : Clear canvas")
print("  ☝️  POINT (index up)     : Undo")
print("\n⌨️  KEYBOARD:")
print("  C : Change color")
print("  + : Increase thickness")
print("  - : Decrease thickness")
print("  S : Save canvas")
print("  Q : Quit")
print(f"\n⚙️  Pinch threshold: {PINCH_THRESHOLD}px (edit PINCH_THRESHOLD in code)")
print("=" * 60 + "\n")

while True:
    success, image = cap.read()
    if not success:
        print("Failed to read frame")
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Initialize canvas
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(image)
        undo_stack.append(drawing_canvas.copy())

    # Process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Cooldown countdown
    if current_cooldown > 0:
        current_cooldown -= 1

    # Hand detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get gesture
        gesture = get_gesture(hand_landmarks, image.shape)

        # Get cursor position
        xc, yc = get_cursor_position(hand_landmarks, image.shape)

        # Smooth cursor movement
        if xp == 0 and yp == 0:
            xp, yp = xc, yc

        xc_smooth = int(xp * (1 - CURSOR_SMOOTHING) + xc * CURSOR_SMOOTHING)
        yc_smooth = int(yp * (1 - CURSOR_SMOOTHING) + yc * CURSOR_SMOOTHING)
        cursor_point = (xc_smooth, yc_smooth)

        # === GESTURE ACTIONS ===

        if gesture == "pinch":
            # DRAW MODE
            cv2.circle(image, cursor_point, 8, (0, 0, 255), cv2.FILLED)

            if not drawing_state:
                # Start new stroke - save canvas to undo stack
                undo_stack.append(drawing_canvas.copy())
                drawing_state = True
                xp, yp = cursor_point

            # Draw line on canvas
            cv2.line(drawing_canvas, (xp, yp), cursor_point,
                     draw_color, line_thickness, cv2.LINE_AA)

            xp, yp = cursor_point

        else:
            # NOT DRAWING
            drawing_state = False
            xp, yp = 0, 0

            # Show cursor (green) if not fist
            if gesture != "fist":
                cv2.circle(image, cursor_point, 10, (0, 255, 0), cv2.FILLED)

            # Handle action gestures (with cooldown)
            if current_cooldown == 0:

                # UNDO ☝️
                if gesture == "point":
                    if len(undo_stack) > 1:
                        undo_stack.pop()
                        drawing_canvas = undo_stack[-1].copy()
                        print("↩️  Undo")
                    current_cooldown = GESTURE_COOLDOWN_FRAMES

                # CLEAR ✌️
                elif gesture == "peace":
                    undo_stack.append(drawing_canvas.copy())
                    drawing_canvas = np.zeros_like(drawing_canvas)
                    undo_stack.append(drawing_canvas.copy())
                    print("🗑️  Clear")
                    current_cooldown = GESTURE_COOLDOWN_FRAMES

    else:
        # No hand detected
        drawing_state = False
        xp, yp = 0, 0

    # === OVERLAY CANVAS ON IMAGE (OPTIMIZED) ===
    # This is the KEY to performance - simple mask overlay
    final_image = image.copy()
    mask = np.any(drawing_canvas != [0, 0, 0], axis=-1)
    final_image[mask] = drawing_canvas[mask]

    # === UI INFO ===
    # Simple info box
    cv2.rectangle(final_image, (0, 0), (200, 80), (50, 50, 50), cv2.FILLED)
    cv2.putText(final_image, 'Color:', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(final_image, (90, 20), 12, draw_color, cv2.FILLED)
    cv2.putText(final_image, f'Thick: {line_thickness}', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS Counter
    frame_count += 1
    if time.time() - fps_time > 1.0:
        fps = frame_count / (time.time() - fps_time)
        frame_count = 0
        fps_time = time.time()

    cv2.putText(final_image, f'FPS: {fps:.0f}', (w - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display
    cv2.imshow('Air Writer - Fast Mode', final_image)

    # === KEYBOARD CONTROLS ===
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        color_index = (color_index + 1) % len(COLORS)
        draw_color = COLORS[color_index]
        print(f"🎨 Color changed")
    elif key == ord('=') or key == ord('+'):
        line_thickness = min(line_thickness + 2, LINE_THICKNESS_MAX)
        print(f"🖌️  Thickness: {line_thickness}")
    elif key == ord('-') or key == ord('_'):
        line_thickness = max(line_thickness - 2, LINE_THICKNESS_MIN)
        print(f"🖌️  Thickness: {line_thickness}")
    elif key == ord('s'):
        # Save canvas
        filename = f"airwriter_{int(time.time())}.png"
        # Save with white background
        white_bg = np.ones_like(drawing_canvas) * 255
        mask = np.any(drawing_canvas != [0, 0, 0], axis=-1)
        save_img = white_bg.copy()
        save_img[mask] = drawing_canvas[mask]
        cv2.imwrite(filename, save_img)
        print(f"💾 Saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Program closed")