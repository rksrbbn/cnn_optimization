# Optimized CNN Image Classifier

Klasifikasi gambar menggunakan **Convolutional Neural Network (CNN)** berbasis **Transfer Learning (MobileNetV2)** dan aplikasi web interaktif dengan **Streamlit** untuk tugas Artificial Intelligence.

---

## [ Instalasi ]

### 1. Clone Repository

```bash
git clone <repository-url>
cd project-folder
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verifikasi Instalasi

```bash
python -c "import tensorflow, streamlit; print('All good!')"
```

---

## [ Persiapan Dataset ]

Struktur folder dataset:

```
dataset/
├── train/
├── validation/
└── test/
```

Pastikan setiap folder berisi subfolder untuk tiap kelas, misalnya:

```
dataset/train/cat/
dataset/train/dog/
```

---

## [ Training Model ]

### Jalankan Training

```bash
python train_optimized_model.py
```

Output model akan tersimpan di folder `outputs/`:

- `optimized_cnn_model.h5`
- `best_model.h5`
- `class_labels.json`
- `training_history.png`

---

## [ Testing & Evaluasi ]

### Test dengan Gambar Baru

Tempatkan gambar di folder:

```
test_images/
```

Lalu jalankan:

```bash
python test_model.py
```

Hasil evaluasi:

- `confusion_matrix.png`
- `test_results_new_images.png`
- Classification report di terminal

---

## [ Menjalankan Aplikasi Web ]

Pastikan model sudah di-train, lalu jalankan:

```bash
streamlit run app.py
```

Akses aplikasi di browser:
[http://localhost:8501](http://localhost:8501)

---

## [ Tools yang digunakan ]

- **TensorFlow** – Model CNN & Transfer Learning
- **Streamlit** – Web app interaktif
- **Plotly** – Visualisasi hasil prediksi

---

## [ Dataset ]

https://www.kaggle.com/datasets/nguynngcan/dataset/data
