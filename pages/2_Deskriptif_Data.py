import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 0. PEMETAAN RENAME FITUR
# ==========================
RENAME_MAP = {
    # pH
    "PH": "pH_tanah",
    "ph": "pH_tanah",
    "pH": "pH_tanah",
    "ph_tanah": "pH_tanah",
    "acidity": "pH_tanah",

    # Suhu
    "temperature": "suhu_tanah",
    "temp": "suhu_tanah",
    "suhu": "suhu_tanah",
    "suhu_tanah": "suhu_tanah",

    # Kelembaban
    "humidity": "kelembaban_udara",
    "humid": "kelembaban_udara",
    "hum": "kelembaban_udara",
    "kelembaban": "kelembaban_udara",
    "kelembaban_udara": "kelembaban_udara",
}


def rename_selected_features(df: pd.DataFrame, selected_features):
    """
    Menyeragamkan nama kolom yang dipilih user menjadi nama standar
    berdasarkan RENAME_MAP.

    Hanya kolom yang ada di selected_features yang dipertimbangkan.
    Kolom lain di df tetap dibiarkan apa adanya.

    Jika terjadi konflik (dua kolom berbeda ingin memakai nama standar yang sama),
    kolom kedua tidak di-rename dan akan diberi peringatan.
    """
    df_renamed = df.copy()
    used_targets = set()
    conflicts = []

    col_rename_dict = {}

    for col in selected_features:
        key = col.lower()
        if key in RENAME_MAP:
            target = RENAME_MAP[key]

            # Jika target sudah dipakai kolom lain
            if target in used_targets and target != col:
                conflicts.append((col, target))
                continue

            # Jika di df sudah ada kolom dengan nama target dan itu bukan kolom yang sama
            if target in df_renamed.columns and target != col:
                conflicts.append((col, target))
                continue

            col_rename_dict[col] = target
            used_targets.add(target)

    if conflicts:
        st.warning(
            "Ada konflik saat penyeragaman nama kolom. "
            "Beberapa kolom tidak diganti karena nama standar sudah dipakai: "
            f"{conflicts}"
        )

    if col_rename_dict:
        df_renamed = df_renamed.rename(columns=col_rename_dict)
        st.info(
            f"Nama kolom berikut diseragamkan secara otomatis: {col_rename_dict}"
        )

    return df_renamed


# ==========================
# CARD PENJELASAN
# ==========================
def explanation_card(title: str, content: str):
    """
    Card penjelasan: title menggunakan HTML box,
    isi menggunakan markdown agar format **bold**, *italic*, dan bullet tetap berfungsi.
    """
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #4CAF50;
            border-radius:6px;
            margin-bottom:0.5rem;
            font-size:15px;
            color:#ffffff;">
            <b>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)


def main():
    st.markdown("## 2. Deskriptif Data")

    # Pastikan data dari halaman 1 sudah tersedia
    if "df_raw" not in st.session_state or st.session_state["df_raw"] is None:
        st.warning(
            "Data belum tersedia. "
            "Silakan kembali ke halaman **1. Pendahuluan & Pemuatan Data** "
            "untuk memuat dataset terlebih dahulu."
        )
        return

    # Pastikan pilihan fitur dari halaman 1 tersedia
    if "selected_features" not in st.session_state or not st.session_state["selected_features"]:
        st.warning(
            "Informasi fitur yang dipilih belum ditemukan. "
            "Kembali ke halaman **1. Pendahuluan & Pemuatan Data** dan pilih minimal tiga fitur "
            "sebelum melanjutkan ke halaman ini."
        )
        return

    df_raw: pd.DataFrame = st.session_state["df_raw"]
    file_name = st.session_state.get("file_name", "(tanpa nama)")
    selected_features = st.session_state["selected_features"]

    # Untuk analisis deskriptif, fokus pada fitur yang dipilih
    df_selected = df_raw[selected_features].copy()

    # ===============================
    # 2.1 INFORMASI UMUM DATASET
    # ===============================
    st.markdown("### 2.1 Informasi Umum Dataset")

    explanation_card(
        "Ringkasan data",
        """
        Bagian ini menampilkan gambaran umum dari data yang sudah dimuat, meliputi:

        - Jumlah baris dan kolom  
        - Nama seluruh kolom di dalam dataset  
        - Fitur yang dipakai untuk analisis pada tahap selanjutnya  
        - Jumlah *missing value* pada seluruh kolom  

        Ringkasan ini membantu memastikan bahwa struktur data sudah sesuai
        sebelum diproses lebih lanjut.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Nama berkas:**", file_name)
        st.write(
            "**Dimensi (seluruh kolom yang diproses):**",
            f"{df_raw.shape[0]} baris × {df_raw.shape[1]} kolom",
        )
    with col2:
        st.write("**Nama semua kolom dalam dataset:**", list(df_raw.columns))
        st.write("**Fitur yang dipilih dari halaman sebelumnya:**", selected_features)
        st.write("**Total missing value (seluruh kolom):**", int(df_raw.isnull().sum().sum()))

    st.write("**Lima baris pertama dataset (seluruh kolom):**")
    st.dataframe(df_raw.head())

    # ===============================
    # 2.2 SEBELUM & SESUDAH PEMILIHAN FITUR
    # ===============================
    st.markdown("---")
    st.markdown("### 2.2 Sebelum & Sesudah Pemilihan Fitur")

    explanation_card(
        "Kenapa perlu pemilihan fitur?",
        """
        Dataset sering berisi banyak kolom, tidak semuanya relevan untuk analisis atau pemodelan.

        Pada tahap sebelumnya, Anda sudah memilih beberapa kolom numerik yang dianggap penting.
        Di sini ditampilkan perbandingan antara:

        - Dataset lengkap (semua kolom)  
        - Dataset yang sudah dipersempit hanya ke kolom yang dipilih sebagai fitur  

        Tujuannya agar jelas kolom mana saja yang benar-benar digunakan pada langkah berikutnya.
        """
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Sebelum pemilihan fitur (seluruh kolom)**")
        st.write(list(df_raw.columns))
        st.dataframe(df_raw.head())

    with colB:
        st.markdown("**Sesudah pemilihan fitur (hanya kolom terpilih)**")
        st.write(selected_features)
        st.dataframe(df_selected.head())

    # ===============================
    # 2.2.1 PENYERAGAMAN NAMA KOLOM FITUR
    # ===============================
    st.markdown("#### 2.2.1 Penyeragaman Nama Kolom Fitur")

    explanation_card(
        "Penyeragaman nama kolom",
        """
        Nama kolom pada file asli bisa berbeda-beda untuk variabel yang mirip.
        Supaya proses di menu berikutnya lebih konsisten, beberapa nama diseragamkan, misalnya:

        - `temperature` / `temp` / `suhu` → `suhu_tanah`  
        - `humidity` / `hum` / `kelembaban` → `kelembaban_udara`  
        - `ph` / `pH` / `acidity` → `pH_tanah`  

        Dengan cara ini, aturan dan model di halaman lain cukup
        mengacu pada nama standar yang sama, tanpa tergantung penamaan di file sumber.
        """
    )

    df_features = rename_selected_features(df_selected, selected_features)

    st.markdown("**Fitur setelah penyeragaman nama kolom:**")
    st.write(list(df_features.columns))
    st.dataframe(df_features.head())

    # Simpan ke session_state untuk digunakan di halaman berikutnya
    st.session_state["df_features"] = df_features

    # ===============================
    # 2.3 MISSING VALUE PER KOLOM
    # ===============================
    st.markdown("---")
    st.markdown("### 2.3 Missing Value per Kolom")

    explanation_card(
        "Pemeriksaan missing value",
        """
        Nilai kosong (*missing value*) bisa muncul karena berbagai hal, misalnya gangguan sensor
        atau data yang tidak tercatat.

        Informasi jumlah dan proporsi *missing value* per kolom berguna untuk:

        - Mengetahui kolom mana yang paling banyak kekosongan datanya  
        - Menentukan strategi imputasi pada tahap preprocessing  
        - Menghindari error ketika data dipakai untuk pemodelan  
        """
    )

    mv = df_features.isnull().sum().to_frame("jumlah_missing")
    mv["proporsi_missing"] = mv["jumlah_missing"] / len(df_features) if len(df_features) > 0 else 0.0
    st.dataframe(mv)

    # ===============================
    # 2.4 STATISTIK DESKRIPTIF
    # ===============================
    st.markdown("---")
    st.markdown("### 2.4 Statistik Deskriptif")

    explanation_card(
        "Ringkasan statistik",
        """
        Statistik deskriptif memberikan gambaran umum mengenai nilai tiap fitur, misalnya:

        - Nilai minimum dan maksimum  
        - Rata-rata (*mean*) dan simpangan baku (*standard deviation*)  
        - Posisi kuartil (Q1, median, Q3)  

        Informasi ini membantu mengecek apakah nilai yang tercatat masih masuk akal
        dan apakah ada indikasi nilai ekstrem yang perlu diperhatikan.
        """
    )

    df_numeric = df_features.select_dtypes(include=[np.number])
    if df_numeric.empty:
        st.warning(
            "Tidak ditemukan kolom numerik pada fitur yang digunakan. "
            "Pastikan setidaknya ada satu fitur numerik yang digunakan."
        )
        return

    st.dataframe(df_numeric.describe().T)

    # ===============================
    # 2.4.1 HISTOGRAM
    # ===============================
    st.markdown("#### 2.4.1 Histogram Tiap Fitur")

    explanation_card(
        "Distribusi nilai fitur",
        """
        Histogram digunakan untuk melihat pola sebaran data pada setiap fitur, misalnya:

        - Apakah nilai menyebar relatif merata atau menumpuk pada rentang tertentu  
        - Apakah distribusi condong ke satu sisi (*skewed*)  
        - Apakah ada nilai yang jauh di luar pola umum (*outlier*)  

        Pengamatan ini bermanfaat sebelum data masuk ke tahap normalisasi dan pemodelan.
        """
    )

    for col in df_numeric.columns:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df_numeric[col].dropna(), bins=20)
            ax.set_title(f"Histogram {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig, use_container_width=True)

        with col_right:
            st.markdown(
                f"""
                **Ringkasan awal untuk `{col}`:**

                - Rentang nilai: `{df_numeric[col].min():.2f}` sampai `{df_numeric[col].max():.2f}`  
                - Rata-rata (mean): `{df_numeric[col].mean():.2f}`  

                Hal yang dapat diperhatikan dari histogram:

                - Sebaran simetris di sekitar tengah → distribusi cenderung normal  
                - Sebaran condong ke kiri/kanan → distribusi *skewed*  
                - Batang yang jauh terpisah dari lainnya → indikasi nilai ekstrem (*outlier*)  
                """
            )

    # ===============================
    # 2.4.2 KORELASI ANTAR FITUR
    # ===============================
    st.markdown("#### 2.4.2 Korelasi Antar Fitur")

    explanation_card(
        "Korelasi fitur",
        """
        Korelasi menunjukkan seberapa kuat hubungan linear antara dua fitur:

        - Nilai mendekati **+1** → hubungan kuat dan searah  
        - Nilai mendekati **−1** → hubungan kuat tetapi berlawanan arah  
        - Nilai mendekati **0** → tidak ada hubungan linear yang jelas  

        Informasi ini dapat digunakan untuk:

        - Mengenali fitur yang sangat mirip (redundan)  
        - Melihat pola hubungan alami antar variabel  
        - Mendeteksi kombinasi fitur yang mungkin perlu diperlakukan khusus  
        """
    )

    if df_numeric.shape[1] >= 2:
        corr = df_numeric.corr()

        col_left, col_right = st.columns(2)

        with col_left:
            fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
            cax = ax_corr.matshow(corr, cmap="coolwarm")
            fig_corr.colorbar(cax)
            ax_corr.set_xticks(range(len(corr.columns)))
            ax_corr.set_yticks(range(len(corr.columns)))
            ax_corr.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=8)
            ax_corr.set_yticklabels(corr.columns, fontsize=8)
            ax_corr.set_title("Matriks Korelasi", fontsize=10, pad=10)
            st.pyplot(fig_corr, use_container_width=True)

        with col_right:
            st.markdown(
                """
                **Cara membaca matriks korelasi secara singkat:**

                - Warna merah → korelasi positif  
                - Warna biru → korelasi negatif  
                - Semakin pekat warnanya → hubungan linear semakin kuat  

                Matriks ini membantu melihat pasangan fitur
                yang bergerak searah atau berlawanan, serta fitur yang hampir tidak berhubungan.
                """
            )

        st.markdown("### Nilai Korelasi Numerik")
        st.dataframe(corr)
    else:
        st.info("Jumlah fitur numerik belum mencukupi untuk perhitungan korelasi.")


if __name__ == "__main__":
    main()
