import cv2
from ultralytics import YOLO
from gesture_detector import GestureDetector
from drawing_canvas import DrawingCanvas


class AirWriter:
    def __init__(self):
        # Load YOLO11n-pose model
        print("Loading YOLO11n-pose model...")
        self.model = YOLO('yolo11n-pose.pt')

        # Initialize gesture detector
        self.gesture_detector = GestureDetector()

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get actual frame size
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Cannot access camera!")

        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")

        # Initialize drawing canvas
        self.canvas = DrawingCanvas(self.frame_width, self.frame_height)

        # State variables
        self.current_mode = "idle"  # idle, drawing, erasing
        self.colors = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 255, 255)  # White
        ]
        self.current_color_idx = 0

        print("Air Writer initialized successfully!")
        print("\nControls:")
        print("- PINCH (jempol + telunjuk): Mulai menggambar")
        print("- OPEN HAND (5 jari terbuka): Hapus canvas")
        print("- C: Ganti warna")
        print("- S: Save canvas")
        print("- Q: Quit")

    def process_frame(self, frame):
        """Process frame dengan YOLO pose detection dan gesture recognition"""
        # Flip frame untuk mirror effect
        frame = cv2.flip(frame, 1)

        # YOLO pose detection
        results = self.model(frame, verbose=False)

        # Annotated frame dari YOLO
        annotated_frame = results[0].plot()

        # Deteksi hand region dari YOLO pose keypoints
        hand_region = self.extract_hand_region(frame, results[0])

        if hand_region is not None:
            hand_img, hand_bbox = hand_region

            # Deteksi landmarks tangan dengan MediaPipe
            hand_landmarks = self.gesture_detector.detect_hand_landmarks(hand_img)

            if hand_landmarks is not None:
                # Adjust landmarks ke koordinat frame asli
                adjusted_landmarks = self.adjust_landmarks_to_frame(
                    hand_landmarks, hand_bbox, frame.shape
                )

                # Deteksi gestures
                self.handle_gestures(adjusted_landmarks, frame.shape)

                # Draw landmarks di frame
                self.draw_landmarks_on_frame(annotated_frame, adjusted_landmarks)

        # Overlay canvas ke frame
        result_frame = self.canvas.overlay_on_frame(annotated_frame, alpha=0.8)

        # Draw UI info
        self.draw_ui(result_frame)

        return result_frame

    def extract_hand_region(self, frame, result):
        """Extract hand region dari YOLO pose keypoints"""
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        keypoints = result.keypoints[0].xy.cpu().numpy()[0]

        # YOLO pose keypoints: 9=left_wrist, 10=right_wrist
        # Ambil pergelangan tangan yang terdeteksi
        wrists = []
        if keypoints[9][0] > 0:  # left wrist
            wrists.append((int(keypoints[9][0]), int(keypoints[9][1])))
        if keypoints[10][0] > 0:  # right wrist
            wrists.append((int(keypoints[10][0]), int(keypoints[10][1])))

        if len(wrists) == 0:
            return None

        # Ambil wrist pertama (bisa dikembangkan untuk multi-hand)
        wrist = wrists[0]

        # Expand region untuk capture seluruh tangan
        expand_size = 200
        x1 = max(0, wrist[0] - expand_size)
        y1 = max(0, wrist[1] - expand_size)
        x2 = min(frame.shape[1], wrist[0] + expand_size)
        y2 = min(frame.shape[0], wrist[1] + expand_size)

        hand_img = frame[y1:y2, x1:x2]

        if hand_img.size == 0:
            return None

        return hand_img, (x1, y1, x2, y2)

    def adjust_landmarks_to_frame(self, hand_landmarks, hand_bbox, frame_shape):
        """Adjust landmark coordinates dari hand region ke frame coordinates"""
        x1, y1, x2, y2 = hand_bbox
        hand_width = x2 - x1
        hand_height = y2 - y1

        adjusted_landmarks = type('obj', (object,), {
            'landmark': []
        })()

        for lm in hand_landmarks.landmark:
            adjusted_lm = type('obj', (object,), {})()
            adjusted_lm.x = (lm.x * hand_width + x1) / frame_shape[1]
            adjusted_lm.y = (lm.y * hand_height + y1) / frame_shape[0]
            adjusted_lm.z = lm.z
            adjusted_landmarks.landmark.append(adjusted_lm)

        return adjusted_landmarks

    def handle_gestures(self, hand_landmarks, image_shape):
        """Handle gesture detection dan actions"""
        # Deteksi pinch untuk drawing
        is_pinching, pinch_point = self.gesture_detector.detect_pinch(
            hand_landmarks, image_shape
        )

        if is_pinching:
            self.current_mode = "drawing"
            self.canvas.draw_line(pinch_point)
        else:
            self.canvas.stop_drawing()

        # Deteksi open hand untuk clear canvas
        is_open = self.gesture_detector.detect_open_hand(
            hand_landmarks, image_shape
        )

        if is_open and self.current_mode != "clearing":
            self.current_mode = "clearing"
            self.canvas.clear_canvas()
            print("Canvas cleared!")
        elif not is_open and self.current_mode == "clearing":
            self.current_mode = "idle"

    def draw_landmarks_on_frame(self, frame, hand_landmarks):
        """Draw hand landmarks pada frame"""
        if hand_landmarks is None:
            return

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        h, w = frame.shape[:2]

        # Draw landmarks
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start_lm = hand_landmarks.landmark[start_idx]
            end_lm = hand_landmarks.landmark[end_idx]

            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))

            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    def draw_ui(self, frame):
        """Draw UI elements (status, controls, dll)"""
        # Background untuk text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text info
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color indicator
        current_color = self.colors[self.current_color_idx]
        cv2.circle(frame, (30, 75), 15, current_color, -1)
        cv2.putText(frame, "Current Color",
                    (55, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls
        cv2.putText(frame, "C: Change Color | S: Save | Q: Quit",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Pinch to Draw | Open Hand to Clear",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def change_color(self):
        """Ganti warna drawing"""
        self.current_color_idx = (self.current_color_idx + 1) % len(self.colors)
        self.canvas.change_color(self.colors[self.current_color_idx])
        print(f"Color changed to index {self.current_color_idx}")

    def run(self):
        """Main loop aplikasi"""
        print("\nStarting Air Writer...")
        print("Please position yourself in front of the camera.\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process frame
                result_frame = self.process_frame(frame)

                # Display
                cv2.imshow('Air Writer - Interactive Board', result_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('c'):
                    self.change_color()
                elif key == ord('s'):
                    self.canvas.save_canvas()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        self.cap.release()
        self.gesture_detector.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    try:
        app = AirWriter()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()