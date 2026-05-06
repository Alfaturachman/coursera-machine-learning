# 📊 Customer Segmentation using RFM & Clustering

## 📝 Deskripsi Proyek
Proyek ini bertujuan untuk mengidentifikasi segmen pelanggan yang berbeda menggunakan teknik **Unsupervised Machine Learning**. Dengan mengelompokkan pelanggan berdasarkan perilaku pembelian mereka, bisnis dapat mengembangkan strategi pemasaran yang lebih personal, meningkatkan retensi pelanggan, dan mengoptimalkan *Customer Lifetime Value* (LTV).

Analisis ini menggunakan metode **RFM (Recency, Frequency, Monetary)** yang merupakan standar industri dalam analisis perilaku pelanggan.

---

## 🎯 Objektif Utama
1.  **Segmentasi Pelanggan:** Mengelompokkan pelanggan ke dalam kategori unik (misal: *Loyal*, *At Risk*, *New Customers*).
2.  **Analisis Karakteristik:** Memahami profil setiap segmen berdasarkan metrik RFM.
3.  **Rekomendasi Strategis:** Memberikan saran tindakan bisnis yang sesuai untuk setiap segmen pelanggan.

---

## 📂 Dataset
Dataset yang digunakan adalah **Online Retail Dataset** dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail).
- **Periode:** 01 Desember 2010 – 09 Desember 2011.
- **Isi:** Transaksi e-commerce non-toko yang berbasis di Inggris.
- **Fitur Utama:** `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

---

## 🛠️ Metodologi & Teknik
Proyek ini mengikuti alur kerja *Data Science* end-to-end:
1.  **Data Cleaning:** Penanganan *missing values*, penghapusan transaksi retur, dan pemfilteran data anomali.
2.  **Feature Engineering:** Transformasi data transaksi menjadi metrik **RFM**:
    - **Recency (R):** Jumlah hari sejak transaksi terakhir.
    - **Frequency (F):** Jumlah total transaksi.
    - **Monetary (M):** Total nilai uang yang dihabiskan.
3.  **Data Preprocessing:** Transformasi log untuk menangani *skewness* dan *Standard Scaling*.
4.  **Model Clustering:**
    - **K-Means Clustering:** Model utama untuk segmentasi.
    - **Elbow Method & Silhouette Score:** Menentukan jumlah klaster (K) optimal.
    - **DBSCAN:** Digunakan untuk mendeteksi outlier dalam perilaku belanja.
    - **Hierarchical Clustering:** Visualisasi struktur kelompok melalui Dendrogram.
5.  **Interpretasi & Visualisasi:** Menggunakan Snake Plot dan Heatmap untuk memahami profil setiap segmen.

---

## 📈 Hasil & Visualisasi
Berikut adalah beberapa hasil visualisasi utama yang dihasilkan dalam analisis ini:

| Visualisasi | Deskripsi |
| :--- | :--- |
| **RFM Distribution** | Melihat sebaran data Recency, Frequency, dan Monetary. |
| **Elbow & Silhouette** | Penentuan jumlah klaster terbaik (K=3 atau K=4). |
| **Cluster Result** | Visualisasi penyebaran pelanggan dalam ruang 3D/2D setelah clustering. |
| **Snake Plot** | Membandingkan karakteristik antar segmen secara relatif. |
| **Business Impact** | Analisis kontribusi pendapatan dan jumlah pelanggan per segmen. |

---

## 🚀 Teknologi yang Digunakan
- **Bahasa:** Python 3.x
- **Libraries:** 
  - `pandas`, `numpy` (Data Manipulation)
  - `matplotlib`, `seaborn` (Data Visualization)
  - `scikit-learn` (Machine Learning: KMeans, DBSCAN, Scaler)
  - `scipy` (Hierarchical Clustering)

---

## 📁 Struktur File
```text
.
├── customer_segmentation_rfm.ipynb  # Notebook utama analisis
├── rfm_with_segments.csv            # Hasil akhir data dengan label segmen
├── cluster_result.png               # Plot hasil clustering
├── dbscan_outlier.png               # Plot deteksi outlier
├── dendrogram.png                   # Plot hierarki clustering
├── elbow_silhouette.png             # Plot evaluasi model
├── rfm_distribution.png             # Plot distribusi fitur RFM
├── segment_business_impact.png      # Plot analisis nilai bisnis
└── snake_plot.png                   # Plot karakteristik segmen
```

---

## ✍️ Penulis
**Alfaturachman**  
*IBM Machine Learning Professional Certificate - Unsupervised Machine Learning*
