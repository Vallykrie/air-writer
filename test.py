import cv2
import mediapipe as mp
import numpy as np
import math

# --- Variabel Konfigurasi ---
LINE_THICKNESS_DEFAULT = 8
LINE_THICKNESS_MIN = 2
LINE_THICKNESS_MAX = 50

GESTURE_COOLDOWN_FRAMES = 30  # Jeda antar gestur (frame)

# ====================================================================
# --- [PERBAIKAN 1] ---
# Threshold statis untuk deteksi pinch (dalam piksel).
# ATUR NILAI INI (misal: 20, 30, 40) sambil testing.
# Semakin kecil angkanya, semakin "rapat" cubitan yang dibutuhkan.
PINCH_THRESHOLD_PIXELS = 10
# ====================================================================


# Daftar warna (BGR format)
COLORS = [
    (255, 255, 0),  # Teal
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 255, 255)  # White
]

# --- Variabel Global Status ---
line_thickness = LINE_THICKNESS_DEFAULT
color_index = 0
draw_color = COLORS[color_index]

xp, yp = 0, 0  # Koordinat x, y *sebelumnya* (untuk smoothing)
drawing_state = False  # Status apakah sedang menggambar
current_cooldown = 0  # Cooldown untuk gestur

# --- Inisialisasi MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# --- Fungsi Bantu (Helper Function) ---

# ====================================================================
# --- [PERBAIKAN 2] ---
# Kita hapus parameter 'hand_size_proxy' karena pinch
# sekarang menggunakan threshold statis.
def get_gesture(hand_landmarks):
    # ====================================================================
    """
    Menganalisis landmark tangan untuk mengenali gestur.
    Mengembalikan string: "pinch", "point", "peace", "open", "fist", "none"
    """
    landmarks = hand_landmarks.landmark

    # 1. Deteksi Jari Terbuka/Tertutup
    # ... (logika jari terbuka/tertutup tetap sama)
    tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    pip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    fingers_up = []
    for i in range(4):  # Loop untuk 4 jari (telunjuk s/d kelingking)
        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
            fingers_up.append(1)  # Jari ke atas
        else:
            fingers_up.append(0)  # Jari ke bawah

    # Format fingers_up: [Index, Middle, Ring, Pinky]

    # 2. Deteksi Gestur Spesifik
    # Gestur "Point" (Undo) ☝️
    if fingers_up == [1, 0, 0, 0]:
        return "point"

    # Gestur "Peace" (Clear) ✌️
    if fingers_up == [1, 1, 0, 0]:
        return "peace"

    # Gestur "Open" (Kursor) 🖐️
    if fingers_up == [1, 1, 1, 1]:
        return "open"

    # Gestur "Fist" (Diam) ✊
    if fingers_up == [0, 0, 0, 0]:
        return "fist"

    # 3. Deteksi Gestur "Pinch" (Menggambar)
    # Ini adalah gestur yang paling penting
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Konversi ke koordinat piksel (meskipun kita hanya perlu jarak)
    h, w, _ = image.shape
    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

    distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

    # ====================================================================
    # --- [PERBAIKAN 3] ---
    # Menggunakan threshold piksel STATIS yang sudah Anda tentukan di atas.
    # INI ADALAH TEMPAT UNTUK TESTING
    if distance < PINCH_THRESHOLD_PIXELS:
        return "pinch"
    # ====================================================================

    # Jika tidak ada gestur di atas, anggap saja kursor
    return "open"  # Default ke mode kursor


# --- Program Utama ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

drawing_canvas = None
undo_stack = []

print("Program berjalan...")
print("  - 'q' : Keluar")
print("  - 'c' : Ganti Warna")
print("  - '+' : Tebalkan Garis")
print("  - '-' : Tipiskan Garis")
print("  - Gestur Pinch : Menggambar")
print("  - Gestur Point ☝️ : Undo")
print("  - Gestur Peace ✌️ : Clear Kanvas")
print(f"  - Threshold Pinch saat ini: {PINCH_THRESHOLD_PIXELS}px (bisa diubah di kode)")

while True:
    success, image = cap.read()
    if not success:
        print("Gagal membaca frame.")
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Inisialisasi kanvas dan tumpukan undo
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(image)
        # Tambahkan kanvas kosong pertama sebagai dasar tumpukan
        undo_stack.append(drawing_canvas.copy())

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Kurangi cooldown setiap frame
    if current_cooldown > 0:
        current_cooldown -= 1

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Gambar kerangka tangan (opsional, bisa dimatikan agar bersih)
        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Fitur Adaptif (Cursor & Pinch) ---
        # Hitung proksi ukuran tangan (jarak pergelangan ke buku jari tengah)
        wrist_lm = hand_landmarks.landmark[0]
        middle_pip_lm = hand_landmarks.landmark[9]

        hand_size_proxy = math.sqrt(
            (wrist_lm.x - middle_pip_lm.x) ** 2 +
            (wrist_lm.y - middle_pip_lm.y) ** 2
        )
        # Normalisasi ke piksel (perkiraan)
        hand_size_px = hand_size_proxy * w

        # Tentukan ukuran kursor adaptif
        # (Logika ini tetap sama, karena kursor adaptif itu bagus)
        cursor_size = int(np.interp(hand_size_px, [50, 200], [20, 8]))
        cursor_size = np.clip(cursor_size, 8, 20)  # Batasi ukuran

        # ====================================================================
        # --- [PERBAIKAN 4] ---
        # Memanggil get_gesture tanpa 'hand_size_proxy'
        gesture = get_gesture(hand_landmarks)
        # ====================================================================

        # Dapatkan koordinat kursor (ujung telunjuk)
        index_tip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        xc, yc = int(index_tip_lm.x * w), int(index_tip_lm.y * h)

        # --- Logika Smoothing Kursor ---
        if xp == 0 and yp == 0:
            # Jika ini frame pertama, set xp, yp ke posisi sekarang
            xp, yp = xc, yc

        # Terapkan smoothing (Linear Interpolation)
        alpha = 0.7
        xc_smooth = int(xp * (1 - alpha) + xc * alpha)
        yc_smooth = int(yp * (1 - alpha) + yc * alpha)
        cursor_point = (xc_smooth, yc_smooth)

        # --- Logika Aksi Berdasarkan Gestur ---

        if gesture == "pinch":
            # Ganti warna kursor menjadi "menggambar" (merah)
            cv2.circle(image, cursor_point, cursor_size, (0, 0, 255), cv2.FILLED)

            # Jika baru mulai menggambar (frame pertama pinch)
            if not drawing_state:
                # Simpan kanvas saat ini ke tumpukan undo
                undo_stack.append(drawing_canvas.copy())
                drawing_state = True
                # Set titik awal gambar ke kursor saat ini
                xp, yp = cursor_point

                # Gambar garis di kanvas
            cv2.line(drawing_canvas, (xp, yp), cursor_point, draw_color, line_thickness)

            # Update titik sebelumnya untuk frame berikutnya
            xp, yp = cursor_point

        else:
            # Jika kita tidak "pinch"
            drawing_state = False  # Set status tidak menggambar
            xp, yp = 0, 0  # Reset titik sebelumnya

            # Tampilkan kursor "bergerak" (hijau)
            if gesture != "fist":  # Sembunyikan kursor jika tangan terkepal
                cv2.circle(image, cursor_point, cursor_size, (0, 255, 0), cv2.FILLED)

            # Cek gestur aksi (jika cooldown selesai)
            if current_cooldown == 0:

                # Gestur "Point" ☝️ (Undo)
                if gesture == "point":
                    print("Gestur terdeteksi: UNDO")
                    if len(undo_stack) > 1:
                        # Hapus kanvas saat ini dari tumpukan
                        undo_stack.pop()
                        # Muat kanvas sebelumnya
                        drawing_canvas = undo_stack[-1].copy()
                    else:
                        print("Tidak ada lagi yang bisa di-undo.")
                    current_cooldown = GESTURE_COOLDOWN_FRAMES  # Set cooldown

                # Gestur "Peace" ✌️ (Clear)
                elif gesture == "peace":
                    print("Gestur terdeteksi: CLEAR")
                    # Simpan kanvas saat ini (sebelum dikosongkan) agar bisa di-undo
                    undo_stack.append(drawing_canvas.copy())
                    # Buat kanvas baru yang kosong
                    drawing_canvas = np.zeros_like(drawing_canvas)
                    # Tambahkan kanvas kosong ini ke tumpukan
                    undo_stack.append(drawing_canvas.copy())
                    current_cooldown = GESTURE_COOLDOWN_FRAMES  # Set cooldown

    else:
        # Jika tidak ada tangan terdeteksi, reset status
        drawing_state = False
        xp, yp = 0, 0

    # --- Tampilkan Hasil ---
    # ====================================================================
    # --- [PERBAIKAN 5] ---
    # Logika baru untuk "menempel" gambar agar SOLID, bukan transparan.

    # 1. Buat salinan gambar video asli
    final_image = image.copy()

    # 2. Buat mask dimana kanvas TIDAK hitam (0,0,0)
    # np.any(...) akan membuat array boolean (True/False)
    mask = np.any(drawing_canvas != [0, 0, 0], axis=-1)

    # 3. "Tempel" kanvas ke gambar asli menggunakan mask
    # Dimana mask bernilai True, piksel di final_image akan
    # diganti dengan piksel dari drawing_canvas.
    final_image[mask] = drawing_canvas[mask]

    # ====================================================================

    # Tampilkan info status (Warna & Ketebalan)
    # Kita gambar info ini di 'final_image' agar selalu di atas
    cv2.rectangle(final_image, (0, 0), (220, 70), (100, 100, 100), cv2.FILLED)
    cv2.putText(final_image, f'Warna:', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(final_image, (100, 25), 15, draw_color, cv2.FILLED)
    cv2.putText(final_image, f'Tebal: {line_thickness}px', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Interactive Whiteboard - (q: quit)', final_image)

    # --- Logika Keyboard ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Ganti warna
        color_index = (color_index + 1) % len(COLORS)
        draw_color = COLORS[color_index]
        print(f"Warna diubah ke: {draw_color}")
    elif key == ord('=') or key == ord('+'):
        # Tambah tebal
        line_thickness = min(line_thickness + 2, LINE_THICKNESS_MAX)
        print(f"Tebal garis: {line_thickness}")
    elif key == ord('-'):
        # Kurangi tebal
        line_thickness = max(line_thickness - 2, LINE_THICKNESS_MIN)
        print(f"Tebal garis: {line_thickness}")

# Bersih-bersih
cap.release()
cv2.destroyAllWindows()