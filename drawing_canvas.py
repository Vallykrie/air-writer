import cv2
import numpy as np
from collections import deque
import config
import copy


class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Buffer untuk smoothing drawing
        self.point_buffer = deque(maxlen=config.SMOOTHING_BUFFER_SIZE)

        # Warna dan ketebalan
        self.draw_color = config.COLORS[config.DEFAULT_COLOR_INDEX]
        self.line_thickness = config.BRUSH_SIZES[config.DEFAULT_BRUSH_INDEX]

        # Status drawing
        self.is_drawing = False
        self.last_point = None

        # Undo history
        self.history = []
        self.save_current_state()  # Save initial empty state

    def save_current_state(self):
        """Save current canvas state untuk undo"""
        if len(self.history) >= config.MAX_UNDO_HISTORY:
            self.history.pop(0)  # Remove oldest

        # Deep copy canvas
        self.history.append(self.canvas.copy())

        if config.VERBOSE:
            print(f"📸 State saved. History size: {len(self.history)}")

    def undo(self):
        """Undo last drawing action"""
        if len(self.history) > 1:  # Keep at least empty canvas
            self.history.pop()  # Remove current state
            self.canvas = self.history[-1].copy()  # Restore previous
            self.stop_drawing()

            if config.VERBOSE:
                print(f"↩️  Undo performed. History size: {len(self.history)}")
            return True
        else:
            if config.VERBOSE:
                print("⚠️  No more undo history")
            return False

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
            if self.is_drawing:
                # Stroke selesai, save state
                self.save_current_state()
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
        else:
            # Starting new stroke
            if not self.is_drawing:
                self.is_drawing = True
                if config.VERBOSE:
                    print("✏️  Started new stroke")

        self.last_point = smoothed_point

    def start_drawing(self):
        """Mulai mode drawing"""
        self.is_drawing = True

    def stop_drawing(self):
        """Stop mode drawing"""
        if self.is_drawing:
            self.save_current_state()
        self.is_drawing = False
        self.last_point = None
        self.point_buffer.clear()

    def clear_canvas(self):
        """Hapus semua gambar di canvas"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stop_drawing()
        self.save_current_state()

        if config.VERBOSE:
            print("🗑️  Canvas cleared")

    def change_color(self, color):
        """Ubah warna drawing"""
        self.draw_color = color

    def change_thickness(self, thickness):
        """Ubah ketebalan garis"""
        self.line_thickness = thickness

    def get_canvas(self):
        """Return canvas saat ini"""
        return self.canvas

    def overlay_on_frame(self, frame, alpha=None):
        """
        Overlay canvas ke frame video dengan transparansi
        alpha: tingkat transparansi (0-1), jika None gunakan config
        """
        if alpha is None:
            alpha = config.CANVAS_ALPHA

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
        """Simpan canvas ke file dengan background putih"""
        # Create white background
        white_bg = np.ones_like(self.canvas) * 255

        # Create mask
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Combine
        bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        result = cv2.add(bg, fg)

        cv2.imwrite(filename, result)
        print(f"💾 Canvas saved to {filename}")
        return filename