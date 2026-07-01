# UAS-KecerdasanBuatan

## Judul Proyek
**Prediksi Rekomendasi Tanaman (Crop Recommendation) Menggunakan Algoritma Descion Tree**

Nama Kelompok:
Wisnu Rangga — 2406168


```

## Penjelasan Langkah Pengerjaan

1. **Persiapan dataset**: Dataset `Crop_recommendation.csv` (sumber Kaggle) diunduh dan disimpan pada folder `data/`. Dataset berisi 7 fitur numerik (N, P, K, temperature, humidity, ph, rainfall) dan 1 target kategorik (`label`) berupa 22 jenis tanaman.
2. **Business & Data Understanding**: Mendefinisikan masalah dunia nyata (kesulitan petani memilih tanaman yang sesuai kondisi lahan) dan memahami struktur data secara menyeluruh (lihat `uas_ai.md` bagian 2-3).
3. **Exploratory Data Analysis (EDA)**: Dilakukan visualisasi distribusi kelas, histogram fitur numerik, heatmap korelasi, pairplot, dan boxplot untuk mendeteksi outlier serta ketidakseimbangan kelas. Kode lengkap ada di `notebook_model.ipynb` bagian 4.
4. **Data Preparation**: Membersihkan data (cek null & duplikat), melakukan label encoding pada target, standardisasi fitur numerik dengan `StandardScaler`, lalu membagi data menjadi data latih (80%) dan data uji (20%) menggunakan stratified split.
5. **Modeling**: Melatih dan membandingkan 4 algoritma klasifikasi: Decision Tree, KNN, SVM, dan Naive Bayes. Decision Tree divisualisasikan sebagai pohon keputusan, dan feature importance dianalisis.
6. **Evaluation**: Mengukur kinerja tiap model menggunakan Accuracy, Precision, Recall, dan F1-score (weighted average), serta menampilkan confusion matrix dan classification report untuk model terbaik.
7. **Kesimpulan**: Menyimpulkan performa model, kelebihan-keterbatasan, dan rekomendasi pengembangan lanjutan.

## Cara Menjalankan
1. Buka `notebook_model.ipynb` di Google Colab (klik "Open in Colab" atau upload manual).
2. Upload file `data/Crop_recommendation.csv` ke environment Colab (atau mount Google Drive dan sesuaikan path).
3. Jalankan seluruh cell secara berurutan (Runtime > Run all).

