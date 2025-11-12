import cv2
import numpy as np
import time
from gesture_detector import MediaPipeGestureDetector
from drawing_canvas import DrawingCanvas
import config


class AirWriterMediaPipe:
    def __init__(self):
        """Initialize Air Writer dengan MediaPipe"""
        print("=" * 60)
        print("🎨 AIR WRITER - MediaPipe Hand Tracking")
        print("=" * 60)

        # Initialize gesture detector
        print("📦 Loading MediaPipe Hands...")
        self.gesture_detector = MediaPipeGestureDetector()
        print("✓ MediaPipe loaded successfully!")

        # Initialize camera
        print(f"📹 Initializing camera {config.CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # Get actual frame size
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("❌ Cannot access camera!")

        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"✓ Camera resolution: {self.frame_width}x{self.frame_height}")

        # Initialize drawing canvas
        self.canvas = DrawingCanvas(self.frame_width, self.frame_height)

        # State variables
        self.current_mode = "idle"  # idle, drawing, clearing
        self.current_color_idx = config.DEFAULT_COLOR_INDEX
        self.current_brush_idx = config.DEFAULT_BRUSH_INDEX

        # Gesture cooldown tracking
        self.last_clear_time = 0
        self.last_undo_time = 0

        # FPS counter
        self.fps = 0
        self.frame_count = 0
        self.fps_update_time = time.time()

        # Print configuration
        self.print_config()

        print("\n" + "=" * 60)
        print("✅ Air Writer initialized successfully!")
        print("=" * 60)
        self.print_controls()

    def print_config(self):
        """Print current configuration"""
        print("\n⚙️  CURRENT CONFIGURATION:")
        print(f"   Pinch Threshold: {config.PINCH_THRESHOLD} pixels")
        print(f"   Peace Separation: {config.PEACE_FINGER_SEPARATION} pixels")
        print(f"   Detection Confidence: {config.MIN_DETECTION_CONFIDENCE}")
        print(f"   Smoothing Buffer: {config.SMOOTHING_BUFFER_SIZE} frames")
        print(f"   Max Undo History: {config.MAX_UNDO_HISTORY} steps")

    def print_controls(self):
        """Print control instructions"""
        print("\n" + "=" * 60)
        print("📝 GESTURE CONTROLS:")
        print("   🤏 PINCH (thumb + index)    : Start drawing")
        print("   ✌️  PEACE SIGN (2 fingers)   : Clear canvas")
        print("   ☝️  POINTING (index up)      : Undo last stroke")
        print("\n⌨️  KEYBOARD CONTROLS:")
        print("   C : Change color")
        print("   B : Change brush size")
        print("   S : Save canvas to file")
        print("   R : Reset/Clear canvas")
        print("   U : Undo last stroke")
        print("   Q : Quit application")
        print("   D : Toggle debug mode")
        print("=" * 60 + "\n")

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_update_time

        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_update_time = current_time

    def process_frame(self, frame):
        """Process frame dengan gesture recognition"""
        # Mirror effect untuk selfie mode
        if config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        # Detect hand landmarks
        hand_landmarks = self.gesture_detector.detect_hand_landmarks(frame)

        if hand_landmarks is not None:
            # Draw landmarks
            self.gesture_detector.draw_landmarks(frame, hand_landmarks)

            # Detect and handle gestures
            self.handle_gestures(hand_landmarks, frame.shape, frame)

            # Draw gesture indicators
            self.gesture_detector.draw_gesture_indicator(frame, hand_landmarks, frame.shape)
        else:
            # No hand detected, stop drawing
            if self.current_mode == "drawing":
                self.canvas.stop_drawing()
                self.current_mode = "idle"

        # Overlay canvas ke frame
        result_frame = self.canvas.overlay_on_frame(frame)

        # Draw UI
        self.draw_ui(result_frame)

        return result_frame

    def handle_gestures(self, hand_landmarks, image_shape, frame):
        """Detect gestures dan handle actions"""
        current_time = time.time()

        # 1. PINCH DETECTION - Drawing
        is_pinching, pinch_point = self.gesture_detector.detect_pinch(
            hand_landmarks, image_shape
        )

        if is_pinching:
            if self.current_mode != "drawing":
                self.current_mode = "drawing"
                if config.VERBOSE:
                    print("✏️  Drawing mode activated")

            self.canvas.draw_line(pinch_point)
        else:
            if self.current_mode == "drawing":
                self.canvas.stop_drawing()
                self.current_mode = "idle"

        # 2. PEACE SIGN DETECTION - Clear Canvas
        if not is_pinching:  # Only check if not pinching
            is_peace = self.gesture_detector.detect_peace_sign(
                hand_landmarks, image_shape
            )

            if is_peace:
                if current_time - self.last_clear_time > config.CLEAR_COOLDOWN:
                    self.canvas.clear_canvas()
                    self.last_clear_time = current_time
                    self.current_mode = "clearing"
                    print("✌️  Peace sign detected - Canvas cleared!")
            else:
                if self.current_mode == "clearing":
                    self.current_mode = "idle"

        # 3. POINTING DETECTION - Undo
        if not is_pinching:  # Only check if not pinching
            is_pointing, point_tip = self.gesture_detector.detect_pointing(
                hand_landmarks, image_shape
            )

            if is_pointing:
                if current_time - self.last_undo_time > config.UNDO_COOLDOWN:
                    if self.canvas.undo():
                        self.last_undo_time = current_time
                        print("☝️  Pointing detected - Undo performed!")

    def draw_ui(self, frame):
        """Draw UI elements"""
        h, w = frame.shape[:2]

        # Main info panel (top-left)
        panel_width = 200
        panel_height = 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        # cv2.putText(frame, "AIR WRITER",
        #             (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mode indicator with color coding
        # mode_color = {
        #     'idle': (200, 200, 200),
        #     'drawing': (0, 255, 0),
        #     'clearing': (0, 0, 255)
        # }
        # cv2.putText(frame, f"Mode: {self.current_mode.upper()}",
        #             (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             mode_color.get(self.current_mode, (255, 255, 255)), 2)

        # Color indicator
        current_color = config.COLORS[self.current_color_idx]
        color_name = config.COLOR_NAMES[self.current_color_idx]
        cv2.circle(frame, (40, 35), 18, current_color, -1)
        cv2.circle(frame, (40, 35), 18, (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {color_name}",
                    (68, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Brush size indicator
        brush_size = config.BRUSH_SIZES[self.current_brush_idx]
        cv2.circle(frame, (40, 75), brush_size, current_color, -1)
        cv2.circle(frame, (40, 75), max(brush_size, 10), (255, 255, 255), 2)
        cv2.putText(frame, f"Brush: {brush_size}px",
                    (68, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Undo history
        # history_count = len(self.canvas.history)
        # cv2.putText(frame, f"Undo History: {history_count}/{config.MAX_UNDO_HISTORY}",
        #             (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # FPS counter (top-right)
        # if config.SHOW_FPS:
        #     fps_text = f"FPS: {self.fps:.1f}"
        #     fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        #     cv2.rectangle(frame, (w - fps_size[0] - 25, 10), (w - 10, 50), (0, 0, 0), -1)
        #     cv2.putText(frame, fps_text, (w - fps_size[0] - 20, 38),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Gesture hints (bottom)
        if config.SHOW_GESTURE_HINTS:
            hint_y = h - 80
            cv2.rectangle(frame, (10, hint_y + 25), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, f"C: Color | B: Brush | S: Save | U: Undo | Q: Quit | PINCH: Draw | PEACE: Clear | POINT: Undo",
                        (20, hint_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Debug info
        if config.DEBUG_MODE:
            debug_y = 250
            cv2.putText(frame, f"PINCH_THRESHOLD: {config.PINCH_THRESHOLD}",
                        (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"PEACE_SEPARATION: {config.PEACE_FINGER_SEPARATION}",
                        (20, debug_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def change_color(self):
        """Cycle through colors"""
        self.current_color_idx = (self.current_color_idx + 1) % len(config.COLORS)
        self.canvas.change_color(config.COLORS[self.current_color_idx])
        color_name = config.COLOR_NAMES[self.current_color_idx]
        print(f"🎨 Color changed to: {color_name}")

    def change_brush_size(self):
        """Cycle through brush sizes"""
        self.current_brush_idx = (self.current_brush_idx + 1) % len(config.BRUSH_SIZES)
        self.canvas.change_thickness(config.BRUSH_SIZES[self.current_brush_idx])
        print(f"🖌️  Brush size changed to: {config.BRUSH_SIZES[self.current_brush_idx]}px")

    def run(self):
        """Main application loop"""
        print("\n🚀 Starting Air Writer...")
        print("📹 Please show your hand to the camera\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break

                # Process frame
                result_frame = self.process_frame(frame)

                # Update FPS
                self.update_fps()

                # Display
                cv2.imshow('Air Writer - MediaPipe Hand Tracking', result_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('c'):
                    self.change_color()
                elif key == ord('b'):
                    self.change_brush_size()
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"air_writer_{timestamp}.png"
                    self.canvas.save_canvas(filename)
                elif key == ord('r'):
                    self.canvas.clear_canvas()
                    print("🗑️  Canvas cleared (keyboard)")
                elif key == ord('u'):
                    if self.canvas.undo():
                        print("↩️  Undo performed (keyboard)")
                elif key == ord('d'):
                    config.DEBUG_MODE = not config.DEBUG_MODE
                    print(f"🐛 Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        self.cap.release()
        self.gesture_detector.release()
        cv2.destroyAllWindows()
        print("✓ Done! Thank you for using Air Writer!\n")


if __name__ == "__main__":
    try:
        app = AirWriterMediaPipe()
        app.run()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback

        traceback.print_exc()