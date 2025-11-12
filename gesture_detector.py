import cv2
import numpy as np
import mediapipe as mp
import config


class MediaPipeGestureDetector:
    """
    Gesture detector menggunakan MediaPipe Hands

    Hand landmarks (21 points):
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.INDEX_MCP = 5
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.MIDDLE_MCP = 9
        self.RING_TIP = 16
        self.RING_MCP = 13
        self.PINKY_TIP = 20
        self.PINKY_MCP = 17

    def detect_hand_landmarks(self, image):
        """Deteksi landmark tangan menggunakan MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def get_landmark_coords(self, landmark, image_shape):
        """Konversi landmark ke koordinat pixel"""
        h, w = image_shape[:2]
        return int(landmark.x * w), int(landmark.y * h)

    def calculate_distance(self, point1, point2):
        """Hitung jarak Euclidean antara 2 titik"""
        if point1 is None or point2 is None:
            return float('inf')
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_finger_extended(self, hand_landmarks, finger_tip_idx, finger_pip_idx, image_shape):
        """Check apakah jari terentang (tip lebih tinggi dari PIP)"""
        tip = self.get_landmark_coords(hand_landmarks.landmark[finger_tip_idx], image_shape)
        pip = self.get_landmark_coords(hand_landmarks.landmark[finger_pip_idx], image_shape)

        # Jari extended jika tip lebih tinggi (y lebih kecil) dari pip
        return tip[1] < pip[1] - 10  # 10px margin

    def detect_pinch(self, hand_landmarks, image_shape):
        """
        🤏 Deteksi gesture PINCH (jempol dan telunjuk menyatu)
        Returns: (is_pinching, pinch_point)

        Customize sensitivity di config.py: PINCH_THRESHOLD
        """
        if hand_landmarks is None:
            return False, None

        # Get thumb tip dan index finger tip
        thumb_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.THUMB_TIP], image_shape
        )
        index_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.INDEX_TIP], image_shape
        )

        # Hitung jarak
        distance = self.calculate_distance(thumb_tip, index_tip)

        if config.DEBUG_MODE:
            print(f"Pinch distance: {distance:.1f} (threshold: {config.PINCH_THRESHOLD})")

        if distance < config.PINCH_THRESHOLD:
            # Return titik tengah untuk drawing
            pinch_point = (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2
            )
            return True, pinch_point

        return False, None

    def detect_peace_sign(self, hand_landmarks, image_shape):
        """
        ✌️ Deteksi gesture PEACE SIGN (telunjuk dan jari tengah terentang, lainnya tertekuk)
        Returns: True if peace sign detected

        Customize sensitivity di config.py: PEACE_FINGER_SEPARATION
        """
        if hand_landmarks is None:
            return False

        # Check index dan middle finger extended
        index_extended = self.is_finger_extended(
            hand_landmarks, self.INDEX_TIP, self.INDEX_PIP, image_shape
        )
        middle_extended = self.is_finger_extended(
            hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, image_shape
        )

        if not (index_extended and middle_extended):
            return False

        # Check ring dan pinky NOT extended (tertekuk)
        ring_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.RING_TIP], image_shape
        )
        ring_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[self.RING_MCP], image_shape
        )
        pinky_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.PINKY_TIP], image_shape
        )
        pinky_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[self.PINKY_MCP], image_shape
        )

        # Ring dan pinky harus tertekuk (tip tidak lebih tinggi dari mcp)
        ring_folded = ring_tip[1] >= ring_mcp[1] - 20
        pinky_folded = pinky_tip[1] >= pinky_mcp[1] - 20

        # Check jarak antara index dan middle finger (harus terpisah)
        index_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.INDEX_TIP], image_shape
        )
        middle_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.MIDDLE_TIP], image_shape
        )

        finger_separation = self.calculate_distance(index_tip, middle_tip)

        if config.DEBUG_MODE:
            print(f"Peace: idx={index_extended}, mid={middle_extended}, "
                  f"ring_fold={ring_folded}, pinky_fold={pinky_folded}, "
                  f"sep={finger_separation:.1f}")

        return (index_extended and middle_extended and
                ring_folded and pinky_folded and
                finger_separation > config.PEACE_FINGER_SEPARATION)

    def detect_pointing(self, hand_landmarks, image_shape):
        """
        ☝️ Deteksi gesture POINTING (hanya telunjuk terentang)
        Returns: (is_pointing, tip_point)

        Customize sensitivity di config.py: FINGER_EXTENSION_RATIO
        """
        if hand_landmarks is None:
            return False, None

        # Get all finger tips dan bases
        wrist = self.get_landmark_coords(hand_landmarks.landmark[self.WRIST], image_shape)

        index_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.INDEX_TIP], image_shape
        )
        index_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[self.INDEX_MCP], image_shape
        )

        middle_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.MIDDLE_TIP], image_shape
        )
        middle_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[self.MIDDLE_MCP], image_shape
        )

        ring_tip = self.get_landmark_coords(
            hand_landmarks.landmark[self.RING_TIP], image_shape
        )
        ring_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[self.RING_MCP], image_shape
        )

        # Check index finger extended
        index_dist_tip = self.calculate_distance(wrist, index_tip)
        index_dist_mcp = self.calculate_distance(wrist, index_mcp)
        index_extended = index_dist_tip > index_dist_mcp * config.FINGER_EXTENSION_RATIO

        # Check other fingers NOT extended
        middle_dist_tip = self.calculate_distance(wrist, middle_tip)
        middle_dist_mcp = self.calculate_distance(wrist, middle_mcp)
        middle_folded = middle_dist_tip < middle_dist_mcp * config.FINGER_EXTENSION_RATIO

        ring_dist_tip = self.calculate_distance(wrist, ring_tip)
        ring_dist_mcp = self.calculate_distance(wrist, ring_mcp)
        ring_folded = ring_dist_tip < ring_dist_mcp * config.FINGER_EXTENSION_RATIO

        if config.DEBUG_MODE:
            print(f"Pointing: idx_ext={index_extended}, mid_fold={middle_folded}, ring_fold={ring_folded}")

        if index_extended and middle_folded and ring_folded:
            return True, index_tip

        return False, None

    def draw_landmarks(self, image, hand_landmarks):
        """Gambar landmark tangan di image"""
        if hand_landmarks and config.SHOW_LANDMARKS:
            self.mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    def draw_gesture_indicator(self, frame, hand_landmarks, image_shape):
        """Draw visual indicators untuk detected gestures"""
        if not config.SHOW_GESTURE_HINTS or hand_landmarks is None:
            return

        # Pinch indicator
        is_pinching, pinch_point = self.detect_pinch(hand_landmarks, image_shape)
        if is_pinching and pinch_point:
            cv2.circle(frame, pinch_point, 15, (0, 255, 255), 3)
            cv2.circle(frame, pinch_point, 5, (0, 255, 255), -1)
            cv2.putText(frame, "PINCH", (pinch_point[0] + 20, pinch_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Peace sign indicator
        if self.detect_peace_sign(hand_landmarks, image_shape):
            wrist = self.get_landmark_coords(hand_landmarks.landmark[self.WRIST], image_shape)
            cv2.putText(frame, "PEACE - CLEAR", (wrist[0] - 50, wrist[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

        # Pointing indicator
        is_pointing, point_tip = self.detect_pointing(hand_landmarks, image_shape)
        if is_pointing and point_tip:
            cv2.circle(frame, point_tip, 12, (255, 0, 255), 3)
            cv2.putText(frame, "UNDO", (point_tip[0] + 20, point_tip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    def release(self):
        """Cleanup resources"""
        self.hands.close()