# Laporan UAS Kecerdasan Buatan

## 1. Judul Proyek
**Prediksi Rekomendasi Tanaman (Crop Recommendation) Menggunakan Algoritma Descion Tree**

Nama Kelompok:
1. Wisnu Rangga — 2406168

---

## 2. Business Understanding

**Permasalahan dunia nyata**
Petani sering mengalami kesulitan dalam menentukan jenis tanaman yang paling sesuai untuk ditanam pada suatu lahan, karena harus mempertimbangkan banyak faktor sekaligus seperti kandungan unsur hara tanah (Nitrogen, Fosfor, Kalium), suhu, kelembapan, pH tanah, dan curah hujan. Kesalahan pemilihan tanaman dapat menyebabkan hasil panen tidak optimal, kerugian ekonomi, dan pemborosan sumber daya (air, pupuk, waktu tanam) (Pudumalar et al., 2017).

**Tujuan proyek**
Membangun model klasifikasi machine learning yang dapat merekomendasikan jenis tanaman paling sesuai berdasarkan parameter tanah dan iklim suatu lahan, sehingga keputusan bercocok tanam menjadi lebih berbasis data (data-driven).

**Siapa user/pengguna sistem**
- Petani dan penyuluh pertanian sebagai pengguna langsung rekomendasi.
- Dinas Pertanian/pemerintah daerah untuk perencanaan distribusi komoditas pertanian.
- Pengembang aplikasi *smart farming* yang ingin mengintegrasikan fitur rekomendasi tanaman.

**Manfaat implementasi AI**
- Meningkatkan produktivitas hasil panen karena tanaman yang dipilih sesuai kondisi lahan.
- Mengurangi risiko gagal panen akibat ketidaksesuaian kondisi tanah/iklim dengan jenis tanaman (Doshi et al., 2018).
- Mendukung praktik pertanian presisi (*precision agriculture*) yang efisien dan berkelanjutan (Kumar et al., 2015).

---

## 3. Data Understanding

**Sumber data**
Dataset *Crop Recommendation Dataset* yang bersumber dari platform Kaggle, berisi kombinasi data agronomi dan iklim yang umum digunakan dalam riset rekomendasi tanaman berbasis machine learning (Pudumalar et al., 2017).

**Deskripsi fitur (atribut)**

| Fitur | Deskripsi | Satuan |
|---|---|---|
| N | Rasio kandungan unsur hara Nitrogen dalam tanah | kg/ha |
| P | Rasio kandungan unsur hara Fosfor dalam tanah | kg/ha |
| K | Rasio kandungan unsur hara Kalium dalam tanah | kg/ha |
| temperature | Suhu rata-rata lingkungan | °C |
| humidity | Kelembapan relatif udara | % |
| ph | Tingkat keasaman/kebasaan tanah | skala 0–14 |
| rainfall | Curah hujan | mm |
| label | Jenis tanaman (target klasifikasi) | kategori |

**Ukuran dan format data**
File CSV (`Crop_recommendation.csv`) berisi ±2200 baris data dan 8 kolom (7 fitur numerik + 1 target kategorik).

**Tipe data dan target klasifikasi**
Seluruh fitur input bertipe numerik kontinu, sedangkan target `label` bertipe kategorik dengan 22 kelas tanaman (klasifikasi multi-kelas), antara lain: rice, maize, jute, cotton, coconut, papaya, orange, apple, muskmelon, watermelon, grapes, mango, banana, pomegranate, lentil, blackgram, mungbean, mothbeans, pigeonpeas, kidneybeans, chickpea, dan coffee.

---

## 4. Exploratory Data Analysis (EDA)

**Visualisasi distribusi data**
Dilakukan visualisasi *bar chart* untuk distribusi jumlah sampel tiap kelas tanaman, dan histogram untuk distribusi tiap fitur numerik (N, P, K, temperature, humidity, ph, rainfall). Hasil visualisasi lengkap tersedia pada `notebook_model.ipynb`.

<img width="1184" height="584" alt="EDA" src="https://github.com/user-attachments/assets/95491ddb-f02c-4ca8-b5c7-a7c4805a59d2" />


**Analisis korelasi antar fitur**
*Heatmap* korelasi Pearson digunakan untuk melihat hubungan linear antar fitur numerik, dan *pairplot* digunakan untuk melihat sebaran beberapa fitur (N, P, K, rainfall) terhadap label pada sampel beberapa jenis tanaman.

<img width="757" height="584" alt="Fitur" src="https://github.com/user-attachments/assets/5f96d719-ef26-4a2b-a664-06e6556f4629" />


**Deteksi data tidak seimbang (imbalanced classes)**
Berdasarkan distribusi jumlah sampel per kelas, dataset ini tergolong **seimbang (balanced)** karena setiap jenis tanaman memiliki jumlah sampel yang relatif sama (sekitar 100 data per kelas), sehingga tidak diperlukan teknik penyeimbangan kelas khusus seperti SMOTE.

**Insight awal dari pola data**
- Setiap jenis tanaman memiliki rentang kebutuhan N, P, K, suhu, kelembapan, pH, dan curah hujan yang relatif khas, sehingga fitur-fitur tersebut informatif untuk membedakan kelas.
- Korelasi antar fitur numerik secara umum tidak terlalu tinggi, sehingga tidak ditemukan indikasi multikolinearitas ekstrem.
- Terdapat beberapa outlier pada fitur seperti *rainfall* dan K, namun hal ini wajar karena mencerminkan kebutuhan ekstrem tanaman tertentu (misalnya tanaman tropis yang membutuhkan curah hujan tinggi).

---

## 5. Data Preparation

**Pembersihan data**
Dilakukan pengecekan nilai kosong (*null value*) dan data duplikat. Baris yang duplikat dihapus menggunakan `drop_duplicates()`, dan baris dengan nilai kosong dihapus menggunakan `dropna()`.

**Encoding data kategorik**
Kolom target `label` (kategorik) diubah menjadi representasi numerik menggunakan *Label Encoding* (`LabelEncoder` dari scikit-learn), karena target bersifat nominal dengan banyak kelas sehingga *label encoding* lebih efisien dibanding *one-hot encoding* untuk target multi-kelas pada model pohon keputusan dan model berbasis jarak.

**Normalisasi/standardisasi data numerik**
Seluruh fitur numerik (N, P, K, temperature, humidity, ph, rainfall) distandardisasi menggunakan `StandardScaler` agar memiliki rata-rata 0 dan standar deviasi 1, sehingga algoritma yang sensitif terhadap skala data (seperti KNN dan SVM) dapat bekerja optimal (Pedregosa et al., 2011).

**Split data (train-test)**
Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dengan parameter `stratify=y` agar proporsi tiap kelas tanaman tetap terjaga pada kedua subset data.

---

## 6. Modeling

**Pemilihan algoritma**
Empat algoritma klasifikasi dibandingkan: **Decision Tree**, **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, dan **Naive Bayes**.

<img width="1570" height="812" alt="Modeling DSC" src="https://github.com/user-attachments/assets/0bbda532-90bb-4312-953d-7aab83021e12" />


**Alasan pemilihan model**
- *Decision Tree* dipilih sebagai model utama karena mudah diinterpretasikan (dapat divisualisasikan sebagai pohon keputusan) dan terbukti efektif untuk masalah klasifikasi tanaman pada penelitian sebelumnya (Doshi et al., 2018).
- *KNN* dipilih sebagai pembanding berbasis kedekatan jarak antar data.
- *SVM* dipilih karena performanya umumnya baik pada klasifikasi multi-kelas dengan margin pemisah yang jelas.
- *Naive Bayes* dipilih sebagai baseline probabilistik yang ringan secara komputasi.

**Implementasi model**
Implementasi lengkap (dengan kode) tersedia pada `notebook_model.ipynb` bagian 6, menggunakan pustaka `scikit-learn` (Pedregosa et al., 2011).

**Visualisasi model**
Pohon keputusan (Decision Tree) divisualisasikan hingga kedalaman 3 level menggunakan `plot_tree`, dan dilengkapi dengan grafik *feature importance* untuk menunjukkan fitur mana yang paling berpengaruh terhadap keputusan klasifikasi.

---

## 7. Evaluation

**Confusion matrix**
Confusion matrix dihitung untuk model dengan performa terbaik guna melihat pola kesalahan klasifikasi antar kelas tanaman (lihat `notebook_model.ipynb` bagian 7.2).

<img width="1297" height="1183" alt="congusion matrix" src="https://github.com/user-attachments/assets/27765da0-a5a6-4068-8f51-5439ea77b6bc" />


**Metrik evaluasi**
Setiap model dievaluasi menggunakan empat metrik utama: **Accuracy**, **Precision**, **Recall**, dan **F1-score** (rata-rata berbobot/*weighted average* karena bersifat multi-kelas).

**Penjelasan kinerja model**
Berdasarkan hasil eksperimen pada notebook, model **Decision Tree** umumnya menghasilkan akurasi, precision, recall, dan F1-score tertinggi atau sangat kompetitif dibanding model lain pada dataset ini (umumnya di atas 95%). Nilai precision dan recall yang tinggi dan seimbang pada hampir seluruh kelas menunjukkan model mampu mengenali pola kebutuhan unsur hara dan iklim tiap tanaman dengan baik, tanpa bias signifikan terhadap kelas tertentu. Kesalahan klasifikasi pada confusion matrix umumnya terjadi pada tanaman dengan karakteristik kondisi tanah/iklim yang mirip (misalnya antar jenis kacang-kacangan seperti mothbeans, mungbean, dan blackgram).



---

## 8. Kesimpulan dan Rekomendasi

**Ringkasan hasil modeling dan evaluasi**
Dari empat algoritma yang diuji (Decision Tree, KNN, SVM, Naive Bayes), model dengan performa terbaik mampu mencapai akurasi tinggi dalam memprediksi jenis tanaman yang sesuai berdasarkan fitur N, P, K, suhu, kelembapan, pH, dan curah hujan.

**Apakah tujuan proyek tercapai?**
Ya. Model klasifikasi yang dibangun mampu memberikan rekomendasi jenis tanaman secara akurat berdasarkan kondisi lahan, sehingga tujuan untuk membantu pengambilan keputusan bercocok tanam berbasis data telah tercapai.

**Kelebihan dan keterbatasan model**
- *Kelebihan*: Decision Tree mudah diinterpretasikan, cepat dilatih, dan akurasinya tinggi pada dataset ini karena pola hubungan fitur-target cukup jelas dan terpisah dengan baik antar kelas.
- *Keterbatasan*: Model berbasis pohon keputusan rentan *overfitting* jika kedalaman tidak dibatasi; dataset hanya mencakup kondisi lahan dari satu wilayah geografis tertentu sehingga generalisasi ke kondisi tanah di Indonesia perlu divalidasi ulang; model belum mempertimbangkan faktor lain seperti jenis tanah, ketersediaan irigasi, dan harga pasar komoditas.

**Rekomendasi perbaikan**
- Mengumpulkan dataset tambahan dari kondisi lahan lokal (Indonesia) untuk meningkatkan relevansi dan generalisasi model.
- Mencoba algoritma *ensemble* seperti Random Forest atau XGBoost untuk meningkatkan akurasi dan mengurangi risiko *overfitting*.
- Melakukan *hyperparameter tuning* (misalnya dengan `GridSearchCV`) untuk mengoptimalkan tiap model.
- Menambahkan fitur lain seperti jenis tanah, ketinggian lahan, dan data historis cuaca musiman untuk meningkatkan akurasi prediksi.

---

## 9. Referensi

1.Apat, S. K., Mishra, J., Raju, K. S., & Padhy, N. (2023). An artificial intelligence-based crop recommendation system using machine learning. Journal of Scientific & Industrial Research, 82(5), 558–567. https://doi.org/10.56042/jsir.v82i05.1092
2.Dey, B., Ferdous, J., & Ahmed, R. (2024). Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables. Heliyon, 10(3), e25112. https://doi.org/10.1016/j.heliyon.2024.e25112
3.Senapaty, M. K., Ray, A., & Padhy, N. (2024). A decision support system for crop recommendation using machine learning classification algorithms. Agriculture, 14(8), 1256. https://doi.org/10.3390/agriculture14081256
4.Pudumalar, S., Ramanujam, E., Rajashree, R. H., Kavya, C., Kiruthika, T., & Nisha, J. (2017). Crop recommendation system for precision agriculture. In 2016 Eighth International Conference on Advanced Computing (ICoAC) (pp. 32–36). IEEE. https://doi.org/10.1109/ICoAC.2017.7951740
5.Attri, I., Awasthi, L. K., & Sharma, T. P. (2024). Machine learning in agriculture: A review of crop management applications. Multimedia Tools and Applications, 83(5), 12875–12915. https://doi.org/10.1007/s11042-023-16105-2

---

## 10. Lampiran

- Dataset mentah: `data/Crop_recommendation.csv`
- Seluruh grafik EDA (histogram, heatmap, pairplot, boxplot), visualisasi pohon keputusan, grafik perbandingan metrik, dan confusion matrix tersedia pada `notebook_model.ipynb`.

