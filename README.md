# 🧠 Prediksi Silent Sufferer — Kesehatan Mental Mahasiswa

Proyek machine learning untuk mendeteksi **Silent Sufferer**: mahasiswa yang mengalami gangguan kesehatan mental (depresi, kecemasan, atau panik) namun **tidak mencari pertolongan profesional**.

---

## 📌 Latar Belakang

Banyak mahasiswa yang diam-diam berjuang dengan masalah kesehatan mental tanpa pernah mencari bantuan. Proyek ini bertujuan membangun model klasifikasi yang dapat mengidentifikasi mahasiswa tersebut, sehingga intervensi dini dapat dilakukan.

---

## 📂 Dataset

**File:** `Student_Mental_health.csv`

| Fitur Asal | Deskripsi |
|---|---|
| `Gender` | Jenis kelamin |
| `Age` | Usia |
| `Course` | Program studi |
| `Year` | Tahun angkatan |
| `CGPA` | Indeks Prestasi Kumulatif |
| `Marital` | Status pernikahan |
| `Depression` | Apakah mengalami depresi (Yes/No) |
| `Anxiety` | Apakah mengalami kecemasan (Yes/No) |
| `Panic` | Apakah mengalami serangan panik (Yes/No) |
| `Treatment` | Apakah sedang mencari pertolongan profesional (Yes/No) |

---

## 🎯 Target

**`Silent_Sufferer`** — mahasiswa yang memiliki setidaknya satu kondisi mental (depresi / kecemasan / panik) **namun tidak** sedang dalam penanganan profesional.

---

## 🔧 Alur Proyek

### 1. Preprocessing
- Load dan inspeksi data awal
- Pembersihan data: rename kolom, hapus duplikat, isi nilai NaN (usia diisi dengan median per tahun angkatan)
- Normalisasi nilai teks (whitespace, kapitalisasi, dll.)

### 2. Exploratory Data Analysis (EDA)
- Distribusi demografi (gender, usia, tahun, status nikah, CGPA, jurusan)
- Prevalensi kondisi mental dan tingkat pencarian pengobatan
- Analisis korelasi antar kondisi mental
- Breakdown kondisi mental berdasarkan gender, tahun, dan CGPA

> **Insight penting:** Hanya sebagian kecil mahasiswa dengan kondisi mental yang aktif mencari pertolongan profesional.

### 3. Feature Engineering

**Grouping jurusan** ke 4 rumpun:
| Rumpun | Contoh Jurusan |
|---|---|
| Engineering & Computing | Engineering, BCS, BIT |
| Natural & Health Sciences | Biomedical Science, Pharmacy, Nursing |
| Social Sciences | Psychology, Economics, Accounting |
| Humanities & Education | Islamic Education, Law, English Language |

**Risk Level** — jumlah kondisi mental yang dimiliki (Low / Medium / High)

**CGPA Grouping:**
- `Below 3.00`
- `3.00 - 3.49`
- `3.50 - 4.00`

**Fitur akhir model:**

| Fitur | Keterangan |
|---|---|
| `Gender` | Boolean (True = Male) |
| `Age` | Numerik |
| `Year` | Tahun angkatan (int) |
| `Marital` | Boolean |
| `CGPA_grouped_encoded` | Ordinal encoded (0–2) |
| `CG_Humanities & Education` | One-Hot Encoding |
| `CG_Natural & Health Sciences` | One-Hot Encoding |
| `CG_Social Sciences` | One-Hot Encoding |

### 4. Modeling

Model yang diujicobakan dengan **StratifiedKFold-5**:

- Baseline (DummyClassifier)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- **XGBoost** ✅

Metrik evaluasi: Accuracy, F1, F1 Macro, **F2**, ROC-AUC

> F2-score diprioritaskan karena **recall lebih penting** — lebih baik false alarm daripada melewatkan mahasiswa yang butuh bantuan.

### 5. Hyperparameter Tuning

GridSearchCV dijalankan untuk semua model utama (LR, RF, SVM, KNN, XGBoost) dengan optimasi terhadap **ROC-AUC**.

### 6. Evaluasi Model Terbaik

Model terbaik: **XGBoost (tuned)**

Evaluasi dilakukan dengan `cross_val_predict` untuk menghasilkan ROC Curve, Precision-Recall Curve, dan Confusion Matrix yang representatif.

### 7. Threshold Optimization

Default threshold 0.5 tidak optimal. Threshold dioptimalkan menggunakan **F2-score** untuk memaksimalkan recall.

> **Threshold optimal:** ~0.28

### 8. Feature Importance

Fitur paling berpengaruh (berdasarkan XGBoost feature importances):
- Jurusan rumpun Natural & Health Sciences
- Status pernikahan (Marital)
- CGPA

---

## 📊 Hasil Akhir

| Metrik | Nilai |
|---|---|
| Model | XGBoost (tuned) |
| ROC-AUC (CV) | ~0.7+ |
| Threshold Optimal | ~0.28 (maximize F2) |
| Strategi | Prioritas recall tinggi (minimize False Negative) |

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install semua dependensi:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 Cara Menjalankan

1. Pastikan file `Student_Mental_health.csv` berada di direktori yang sama dengan notebook.
2. Buka file `tekra.ipynb` di Jupyter Notebook / JupyterLab / VS Code.
3. Jalankan semua cell secara berurutan dari atas ke bawah.

---

## 📝 Catatan

- Dataset memiliki ketidakseimbangan kelas (*imbalanced*); model menggunakan `class_weight='balanced'` dan `scale_pos_weight` (XGBoost) untuk menanganinya.
- Threshold rendah (~0.28) disengaja agar recall tinggi — sesuai konteks masalah di mana melewatkan seorang Silent Sufferer lebih berbahaya dibandingkan false alarm.
