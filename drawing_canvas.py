import cv2
import numpy as np
from collections import deque


class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Buffer untuk smoothing drawing
        self.point_buffer = deque(maxlen=5)

        # Warna dan ketebalan
        self.draw_color = (0, 255, 255)  # Cyan
        self.line_thickness = 3

        # Status drawing
        self.is_drawing = False
        self.last_point = None

    def add_point(self, point):
        """Tambah titik ke buffer untuk smoothing"""
        if point is not None:
            self.point_buffer.append(point)

    def get_smoothed_point(self):
        """Dapatkan titik yang sudah di-smooth menggunakan moving average"""
        if len(self.point_buffer) == 0:
            return None

        avg_x = int(np.mean([p[0] for p in self.point_buffer]))
        avg_y = int(np.mean([p[1] for p in self.point_buffer]))
        return (avg_x, avg_y)

    def draw_line(self, point):
        """Gambar garis dari last_point ke point saat ini"""
        if point is None:
            self.is_drawing = False
            self.last_point = None
            self.point_buffer.clear()
            return

        self.add_point(point)
        smoothed_point = self.get_smoothed_point()

        if smoothed_point is None:
            return

        if self.is_drawing and self.last_point is not None:
            cv2.line(
                self.canvas,
                self.last_point,
                smoothed_point,
                self.draw_color,
                self.line_thickness,
                cv2.LINE_AA
            )

        self.is_drawing = True
        self.last_point = smoothed_point

    def start_drawing(self):
        """Mulai mode drawing"""
        self.is_drawing = True

    def stop_drawing(self):
        """Stop mode drawing"""
        self.is_drawing = False
        self.last_point = None
        self.point_buffer.clear()

    def clear_canvas(self):
        """Hapus semua gambar di canvas"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stop_drawing()

    def change_color(self, color):
        """Ubah warna drawing"""
        self.draw_color = color

    def get_canvas(self):
        """Return canvas saat ini"""
        return self.canvas

    def overlay_on_frame(self, frame, alpha=0.7):
        """
        Overlay canvas ke frame video dengan transparansi
        alpha: tingkat transparansi (0-1)
        """
        # Buat mask dari canvas (area yang ada gambar)
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)

        # Inverse mask
        mask_inv = cv2.bitwise_not(mask)

        # Ambil region dari frame dimana tidak ada gambar
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Ambil gambar dari canvas
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        # Blend dengan alpha
        canvas_fg_weighted = cv2.addWeighted(canvas_fg, alpha, np.zeros_like(canvas_fg), 0, 0)

        # Combine
        result = cv2.add(frame_bg, canvas_fg_weighted)

        return result

    def save_canvas(self, filename="air_writer_output.png"):
        """Simpan canvas ke file"""
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved to {filename}")