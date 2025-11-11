import cv2
import numpy as np
import mediapipe as mp


class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

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
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def detect_pinch(self, hand_landmarks, image_shape):
        """
        Deteksi gesture pinch (jempol dan telunjuk menyatu)
        Return: (is_pinching, pinch_point)
        """
        if hand_landmarks is None:
            return False, None

        # Index finger tip (landmark 8) dan Thumb tip (landmark 4)
        index_tip = self.get_landmark_coords(
            hand_landmarks.landmark[8], image_shape
        )
        thumb_tip = self.get_landmark_coords(
            hand_landmarks.landmark[4], image_shape
        )

        # Hitung jarak
        distance = self.calculate_distance(index_tip, thumb_tip)

        # Threshold untuk pinch (sesuaikan jika perlu)
        pinch_threshold = 40

        if distance < pinch_threshold:
            # Return titik tengah antara jempol dan telunjuk
            pinch_point = (
                (index_tip[0] + thumb_tip[0]) // 2,
                (index_tip[1] + thumb_tip[1]) // 2
            )
            return True, pinch_point

        return False, None

    def detect_open_hand(self, hand_landmarks, image_shape):
        """
        Deteksi gesture tangan terbuka (seperti high-five)
        Cek apakah semua jari terentang
        """
        if hand_landmarks is None:
            return False

        # Landmark tips jari: thumb(4), index(8), middle(12), ring(16), pinky(20)
        # Landmark base jari: thumb(2), index(5), middle(9), ring(13), pinky(17)
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]

        fingers_extended = 0

        # Cek setiap jari (kecuali jempol)
        for i in range(1, 5):
            tip = self.get_landmark_coords(
                hand_landmarks.landmark[finger_tips[i]], image_shape
            )
            base = self.get_landmark_coords(
                hand_landmarks.landmark[finger_bases[i]], image_shape
            )

            # Jari terentang jika tip lebih tinggi dari base (y lebih kecil)
            if tip[1] < base[1]:
                fingers_extended += 1

        # Jempol cek horizontal (x direction)
        thumb_tip = self.get_landmark_coords(
            hand_landmarks.landmark[4], image_shape
        )
        thumb_base = self.get_landmark_coords(
            hand_landmarks.landmark[2], image_shape
        )

        if abs(thumb_tip[0] - thumb_base[0]) > abs(thumb_tip[1] - thumb_base[1]):
            fingers_extended += 1

        # Tangan terbuka jika minimal 4 jari terentang
        return fingers_extended >= 4

    def detect_pointing(self, hand_landmarks, image_shape):
        """
        Deteksi gesture pointing (hanya telunjuk terentang)
        Return: (is_pointing, tip_point)
        """
        if hand_landmarks is None:
            return False, None

        # Cek telunjuk terentang
        index_tip = self.get_landmark_coords(
            hand_landmarks.landmark[8], image_shape
        )
        index_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[5], image_shape
        )

        # Cek jari tengah tertekuk
        middle_tip = self.get_landmark_coords(
            hand_landmarks.landmark[12], image_shape
        )
        middle_mcp = self.get_landmark_coords(
            hand_landmarks.landmark[9], image_shape
        )

        index_extended = index_tip[1] < index_mcp[1] - 20
        middle_folded = middle_tip[1] > middle_mcp[1]

        if index_extended and middle_folded:
            return True, index_tip

        return False, None

    def draw_landmarks(self, image, hand_landmarks):
        """Gambar landmark tangan di image"""
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    def release(self):
        """Cleanup resources"""
        self.hands.close()