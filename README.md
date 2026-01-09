# Deteksi Komentar Toksik pada Chat Game (Hybrid CNN-LSTM + Tomek Links)

Repository ini merupakan implementasi kode untuk **Project** dengan judul:
**"Implementasi Metode Hybrid CNN-LSTM dan Tomek Links untuk Deteksi Komentar Toksik pada Chat Game"**.

Aplikasi ini dibangun menggunakan **Python** dan framework **Deep Learning (TensorFlow/Keras)**, serta dilengkapi dengan antarmuka web berbasis **Streamlit** untuk demonstrasi deteksi secara *real-time*.

## 📋 Fitur Utama
* **Preprocessing Text:** Pembersihan simbol, angka, URL, dan *stemming* bahasa Indonesia (Sastrawi).
* **Imbalanced Data Handling:** Menggunakan **Tomek Links** untuk menyeimbangkan data kelas mayoritas dan minoritas.
* **Hybrid Architecture:** Menggabungkan **CNN** (untuk ekstraksi fitur lokal) dan **LSTM** (untuk menangkap konteks kalimat panjang).
* **Interactive Web App:** Antarmuka pengguna sederhana yang otomatis menyesuaikan tema sistem (Light/Dark Mode).

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-learn, Imbalanced-learn
* **NLP:** Sastrawi (Indonesian Stemmer)

## 📂 Struktur Folder
* `app.py`: File utama aplikasi web (Streamlit).
* `model_toxic_game.h5`: Model Deep Learning yang sudah dilatih.
* `tokenizer.pickle`: File tokenizer untuk konversi teks ke urutan angka.
* `indonesian_chat.csv`: Dataset chat game berbahasa Indonesia.

## 🚀 Cara Menjalankan (Local)

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/agustiofathoni/Implementasi-Hybrid-CNN-LSTM-dan-Tomek-Links-untuk-Deteksi-Komentar-Toksik-pada-Chat-Game.git](https://github.com/agustiofathoni/Implementasi-Hybrid-CNN-LSTM-dan-Tomek-Links-untuk-Deteksi-Komentar-Toksik-pada-Chat-Game.git)
    cd Implementasi-Hybrid-CNN-LSTM-dan-Tomek-Links-untuk-Deteksi-Komentar-Toksik-pada-Chat-Game
    ```

2.  **Buat Virtual Environment** (Disarankan Python 3.11 untuk stabilitas TensorFlow)
    ```bash
    # Mac/Linux
    python3.11 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Khusus pengguna Mac M1/M2/M3, pastikan menggunakan environment Python 3.10 atau 3.11 agar TensorFlow berjalan lancar)*.

4.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

## 👤 Author
**Kelompok 8**
* Mahasiswa Informatika - Universitas Internasional Semen Indonesia (UISI)
* Project Deep Learning Semester 7

---