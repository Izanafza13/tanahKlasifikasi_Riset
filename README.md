# 🌾 Klasifikasi Kesesuaian Lahan Padi (Rice Land Suitability)

Dashboard Streamlit untuk mengklasifikasikan kesesuaian lahan sawah untuk padi berdasarkan **pH tanah**, **suhu tanah**, dan **kelembaban udara**. Model yang tersedia: Random Forest, MLP, Hybrid Soft Voting (RF + MLP), dan Hybrid Stacking (RF + MLP + LR), lengkap dengan interpretabilitas (SHAP/LIME) dan prediksi interaktif.

## Isi Repository

| File / Folder | Keterangan |
|---|---|
| `Beranda.py` | Halaman utama (entry point aplikasi) |
| `pages/1_Pendahuluan_Load_Data.py` | Upload CSV dan pilih fitur |
| `pages/2_Deskriptif_Data.py` | Statistik deskriptif data |
| `pages/3_Distribusi_Label_Imputasi.py` | Imputasi KNN dan pelabelan rule-based |
| `pages/4_Split_SMOTE_Normalisasi.py` | Split data, SMOTE, normalisasi |
| `pages/5_Model_Evaluasi.py` | Pelatihan dan evaluasi model |
| `pages/6_Interpretabilitas_Model.py` | SHAP dan LIME |
| `pages/7_Prediksi_Interaktif.py` | Prediksi dari input manual |
| `requirements.txt` | Daftar library beserta versinya |

## Persyaratan

- Python 3.10 atau lebih baru
- Git (opsional, untuk clone repository)

## Instalasi

1. **Clone repository**

   ```
   git clone https://github.com/Izanafza13/tanahKlasifikasi_Riset.git
   cd tanahKlasifikasi_Riset
   ```

2. **Buat virtual environment** (disarankan agar versi library tidak bentrok dengan proyek lain)

   ```
   python -m venv .venv
   ```

3. **Aktifkan virtual environment**

   - Windows (Command Prompt): `.venv\Scripts\activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS / Linux: `source .venv/bin/activate`

   > Jika PowerShell menolak, jalankan sekali: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

4. **Install library**

   ```
   python -m pip install -U pip
   python -m pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

```
python -m streamlit run Beranda.py
```

Browser akan terbuka otomatis di `http://localhost:8501`. Jika tidak, buka alamat tersebut secara manual. Untuk menghentikan aplikasi, tekan `Ctrl + C` di terminal.

## Format Dataset

Siapkan file **CSV** dengan kolom numerik. Agar pelabelan rule-based berjalan, gunakan **nama kolom persis** seperti berikut:

| Kolom | Keterangan |
|---|---|
| `pH_tanah` | pH tanah |
| `suhu_tanah` | Suhu tanah (°C) |
| `kelembaban_udara` | Kelembaban udara (%) |

Catatan:
- Kolom `label` (jika ada) dan kolom waktu (tahun, tanggal, dan sejenisnya) akan diabaikan otomatis.
- Pilih **minimal 3 kolom** sebagai fitur.
- Nilai kosong (missing value) diisi otomatis dengan KNN Imputer (k = 3).
- Dataset tidak disertakan di repository ini; unggah file milik Anda sendiri.

## Tata Cara Penggunaan

Ikuti halaman secara **berurutan** dari menu sidebar kiri. Setiap halaman memakai hasil dari halaman sebelumnya, jadi jangan melompat atau me-refresh browser di tengah proses (data tersimpan di sesi dan akan hilang saat refresh).

1. **Pendahuluan & Load Data**: unggah file CSV, periksa ringkasan data, lalu centang kolom yang dipakai sebagai fitur (minimal 3).
2. **Deskriptif Data**: lihat statistik dan sebaran data fitur yang dipilih.
3. **Distribusi Label & Imputasi**: nilai kosong diimputasi otomatis, lalu data diberi label `label_lahan`:
   - `2` = Sangat Cocok
   - `1` = Kurang Cocok
   - `0` = Tidak Cocok

   Lihat juga distribusi tiap kelas.
4. **Split, SMOTE & Normalisasi**: atur proporsi data uji dengan slider, pilih apakah memakai SMOTE untuk menyeimbangkan kelas, lalu data dinormalisasi otomatis.
5. **Model & Evaluasi**: pilih model (Random Forest, MLP, Hybrid Soft Voting, atau Hybrid Stacking), atur hyperparameter, lalu klik **🚀 Latih Model Terpilih**. Hasil evaluasi (akurasi, confusion matrix, dan metrik lain) akan tampil.
6. **Interpretabilitas Model**: lihat fitur yang paling berpengaruh (SHAP) dan penjelasan per sampel (LIME). Halaman ini memakai model terakhir yang dilatih di halaman 5.
7. **Prediksi Interaktif**: masukkan nilai pH tanah, suhu tanah, dan kelembaban udara untuk memprediksi kelas kesesuaian lahan dengan model yang sudah dilatih.

## Troubleshooting

- **`ImportError: cannot import name '_get_column_indices' from 'sklearn.utils'`**: versi `imbalanced-learn` tidak cocok dengan `scikit-learn`. Install ulang sesuai `requirements.txt` di virtual environment baru, atau jalankan `python -m pip install -U imbalanced-learn`.
- **Halaman menampilkan peringatan "belum ada data/model"**: ada langkah sebelumnya yang belum dijalankan. Kembali ke halaman sebelumnya dan selesaikan dulu.
- **`streamlit` tidak dikenali**: jalankan dengan `python -m streamlit run Beranda.py` dan pastikan virtual environment sudah aktif.

## Lisensi

Apache License 2.0, lihat berkas [LICENSE](LICENSE).
