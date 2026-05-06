# Modul 1: Pendahuluan Jaringan Saraf Tiruan (Introduction to Neural Networks)

Selamat datang di direktori Modul 1. Folder ini berisi materi pembelajaran mengenai dasar-dasar Jaringan Saraf Tiruan (Neural Networks) dan algoritma optimasi yang mendasarinya. Seluruh materi telah disesuaikan ke dalam Bahasa Indonesia untuk memudahkan proses belajar.

## Daftar Isi Materi

Berikut adalah ringkasan dari tiga notebook utama yang dipelajari dalam modul ini:

### 1. [05a_DEMO_Gradient_Descent.ipynb](./05a_DEMO_Gradient_Descent.ipynb)

Notebook ini berfokus pada **Gradient Descent**, yaitu algoritma optimasi paling populer yang digunakan untuk melatih model Machine Learning dan Deep Learning.

- **Topik Utama:**
    - Visualisasi lintasan pencarian parameter optimal.
    - Efek dari _Learning Rate_ (terlalu besar vs terlalu kecil).
    - Perbedaan antara **Batch Gradient Descent** dan **Stochastic Gradient Descent (SGD)**.
    - Pentingnya pengacakan data (_shuffling_) dalam proses pelatihan.

### 2. [05b_LAB_Intro_NN.ipynb](./05b_LAB_Intro_NN.ipynb)

Notebook ini memperkenalkan konsep dasar unit terkecil dari jaringan saraf, yaitu **Neuron** (atau Perceptron).

- **Topik Utama:**
    - Menggunakan neuron tunggal sebagai **Gerbang Logika Boolean** (AND, OR, NAND, NOR).
    - Memahami batasan linearitas (kasus gerbang XOR).
    - Representasi jaringan saraf sebagai rangkaian **Komputasi Matriks** (perkalian bobot dan fungsi aktivasi).

### 3. [NN_with_Sklearn.ipynb](./NN_with_Sklearn.ipynb)

Notebook ini menerapkan konsep jaringan saraf pada kasus nyata menggunakan pustaka standar industri, **scikit-learn**.

- **Topik Utama:**
    - Implementasi **Multi-layer Perceptron (MLP)**.
    - Pengenalan angka tulisan tangan (Dataset Digit Recognition).
    - Teknik optimasi hiperparameter menggunakan `RandomizedSearchCV`.
    - Evaluasi performa model menggunakan _Confusion Matrix_ dan skor akurasi.

---

## Persyaratan (Prerequisites)

Untuk menjalankan notebook di atas, pastikan Anda telah menginstal pustaka Python berikut:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Cara Penggunaan

Silakan buka file `.ipynb` secara berurutan mulai dari demo (05a) hingga implementasi praktis (NN_with_Sklearn) menggunakan Jupyter Notebook atau VS Code Jupyter Extension.

---

_Catatan: Seluruh dokumentasi markdown dan komentar kode telah diperbarui untuk memastikan pemahaman konsep yang lebih baik._
