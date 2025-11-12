# 🎨 Air Writer: MediaPipe Hand Tracking Drawing Application

Air Writer adalah aplikasi *Computer Vision* yang memungkinkan pengguna untuk menggambar di atas kanvas virtual secara *real-time* hanya dengan menggunakan gerakan tangan, yang dideteksi melalui *webcam*. Proyek ini memanfaatkan model *pose estimation* canggih dari MediaPipe untuk melacak *landmark* tangan dan mengenali *gesture* (gerakan) spesifik.

Proyek ini dibangun menggunakan **Python** dan **OpenCV**.

## ✨ Fitur Utama

* **Real-time Hand Tracking:** Menggunakan MediaPipe untuk mendeteksi 21 *landmark* tangan dengan latensi rendah.
* **Gesture-based Drawing:** Gerakan **Pinch (cubit)** (jempol dan telunjuk menyatu) untuk mulai menggambar.
* **Gesture Controls:**
    * ✌️ **Peace Sign (jari tengah & telunjuk):** Membersihkan (Clear) kanvas.
    * ☝️ **Pointing (telunjuk terentang):** Membatalkan (Undo) goresan terakhir.
* **Customizable Brush:** Dapat mengubah warna dan ukuran kuas melalui *keyboard* atau konfigurasi.
* **Smooth Drawing:** Implementasi *Mean Filter* pada koordinat kursor untuk menghasilkan garis yang lebih halus.
* **Canvas Overlay:** Menggabungkan gambar yang digambar dengan *feed* kamera menggunakan teknik *blending* dan *masking*.

## 🚀 Instalasi dan Penggunaan

### 1. Prasyarat

Pastikan Anda telah menginstal **Python 3.7+**.

### 2. Instalasi Dependencies

Instal semua *library* yang diperlukan menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt