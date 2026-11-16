import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer  # <--- TAMBAHAN IMPORT


# ==============================
# 1. CARD PENJELASAN
# ==============================
def explanation_card(title: str, content: str):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #2196F3;
            border-radius:6px;
            margin-bottom:0.5rem;
            color:#ffffff;">
            <b>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)


# ==============================
# 2. RULE-BASED (3 FITUR INTI)
# ==============================
def label_kesesuaian_lahan(row):
    """
    Rule-based kesesuaian lahan padi berdasarkan 3 fitur utama:

      - pH_tanah
      - suhu_tanah
      - kelembaban_udara

    Jika salah satu fitur tidak tersedia → dilewati.
    """

    statuses = []

    # -------------------- pH tanah --------------------
    if "pH_tanah" in row.index:
        pH = row["pH_tanah"]
        if pH < 4.0 or pH > 8.5:
            status_pH = 0
        elif 5.5 <= pH <= 7.0:
            status_pH = 2
        elif (4.5 <= pH < 5.5) or (7.0 < pH <= 8.0):
            status_pH = 1
        else:
            status_pH = 1
        statuses.append(status_pH)

    # -------------------- Suhu --------------------
    if "suhu_tanah" in row.index:
        suhu = row["suhu_tanah"]
        if suhu < 18 or suhu > 35:
            status_T = 0
        elif 24 <= suhu <= 29:
            status_T = 2
        elif (22 <= suhu < 24) or (29 < suhu <= 32):
            status_T = 1
        else:
            status_T = 1
        statuses.append(status_T)

    # -------------------- Kelembaban --------------------
    if "kelembaban_udara" in row.index:
        kelembaban = row["kelembaban_udara"]
        if kelembaban < 30:
            status_RH = 0
        elif 33 <= kelembaban <= 90:
            status_RH = 2
        else:
            status_RH = 1
        statuses.append(status_RH)

    # Jika kosong (sangat jarang)
    if len(statuses) == 0:
        return 1

    # Hukum minimum
    if 0 in statuses:
        return 0
    elif all(s == 2 for s in statuses):
        return 2
    else:
        return 1


# ==============================
# 3. MENU UTAMA
# ==============================
def main():

    st.markdown("## 3. Distribusi Label & Imputasi")

    # Ambil data dari Page 2
    if "df_features" in st.session_state and st.session_state["df_features"] is not None:
        df = st.session_state["df_features"].copy()
    elif "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
        df = st.session_state["df_raw"].copy()
    else:
        st.warning(
            "Belum terdapat data yang dimuat. Muat dataset terlebih dahulu pada "
            "**1. Pendahuluan & Pemuatan Data**."
        )
        return

    # ==============================
    # 3.1 IMPUTASI MISSING VALUE
    # ==============================
    st.markdown("### 3.1 Imputasi Missing Value")

    explanation_card(
        "Alasan dilakukan imputasi",
        """
        Nilai kosong (*missing value*) bisa muncul akibat gangguan sensor, 
        pencatatan yang tidak lengkap, atau kesalahan input.

        Jika dibiarkan, baris yang memiliki nilai kosong sering kali harus dibuang,
        sehingga jumlah data berkurang dan informasi terbuang percuma.

        Di sini digunakan **KNN Imputer (K-Nearest Neighbors, k=3)**:
        - Setiap nilai kosong akan diisi berdasarkan rata-rata nilai dari
          **3 tetangga terdekat** (baris lain yang mirip) untuk kolom tersebut.
        - Pendekatan ini lebih kontekstual dibanding sekadar pakai satu angka global,
          karena mempertimbangkan pola data di sekitarnya.

        Dengan cara ini, jumlah sampel tetap terjaga dan pola hubungan antar fitur
        lebih realistis untuk dipakai melatih model.
        """
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # >>> GANTI MEAN → KNN IMPUTER <<<
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    st.markdown("**Dataset setelah imputasi (contoh 5 baris pertama):**")
    st.dataframe(df.head())

    # ==============================
    # 3.2 PELABELAN RULE-BASED
    # ==============================
    st.markdown("---")
    st.markdown("### 3.2 Pelabelan Rule-Based")

    explanation_card(
        "Dasar pelabelan rule-based",
        """
        Pelabelan kesesuaian lahan dilakukan berdasarkan **tiga variabel utama** 
        yang paling berpengaruh dalam pertumbuhan padi:

        - **pH tanah** → menentukan tingkat keasaman tanah  
        - **suhu tanah** → mempengaruhi aktivitas metabolisme tanaman  
        - **kelembaban udara** → berpengaruh terhadap penguapan dan kenyamanan mikroklimat  

        Setiap variabel diberi kategori numerik:

        - `2` → **Sangat Cocok**  
        - `1` → **Kurang Cocok**  
        - `0` → **Tidak Cocok**  

        Hasil akhir label lahan mengikuti prinsip *faktor pembatas*:

        - Jika **satu saja** dari ketiga faktor bernilai `0` (Tidak Cocok), 
          maka lahan secara keseluruhan diberi label `0`.  
        - Jika **semua** faktor bernilai `2` (Sangat Cocok), maka label akhir `2`.  
        - Selain itu (kombinasi campuran 1 dan 2), lahan diberi label `1` (Kurang Cocok).  

        Dengan demikian, label akhir `label_lahan` juga berupa **3 kelas**:
        `0`, `1`, dan `2`, yang akan dipakai sebagai target pada model klasifikasi.
        """
    )

    df["label_lahan"] = df.apply(label_kesesuaian_lahan, axis=1)

    st.markdown("**Preview data dengan label (5 baris pertama):**")
    st.dataframe(df.head())

    st.session_state["df_labeled"] = df

    # ==============================
    # 3.3 DISTRIBUSI LABEL
    # ==============================
    st.markdown("---")
    st.markdown("### 3.3 Distribusi Label")

    explanation_card(
        "Tujuan analisis distribusi label",
        """
        Distribusi label menunjukkan kondisi sebaran antara tiga kategori utama:

        - **0 — Tidak Cocok**  
        - **1 — Kurang Cocok**  
        - **2 — Sangat Cocok**

        Hal ini penting karena:

        - Jika salah satu kelas jumlahnya jauh lebih banyak,
          model cenderung **bias** ke kelas tersebut.  
        - Jika kelas tertentu sangat sedikit, model akan **sulit belajar pola** untuk kelas itu.  

        Informasi ini akan menjadi dasar mengapa pada tahap berikutnya
        digunakan teknik seperti **SMOTE** untuk menyeimbangkan jumlah data per kelas.
        """
    )

    label_counts = df["label_lahan"].value_counts().sort_index()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ["red", "orange", "green"]
        ax.bar(label_counts.index, label_counts.values,
               color=[colors[i] for i in range(len(label_counts))])
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("Label")
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Label")
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown(
            f"""
            **Jumlah data per kelas:**  

            - **0 (Tidak Cocok):** {label_counts.get(0, 0)}  
            - **1 (Kurang Cocok):** {label_counts.get(1, 0)}  
            - **2 (Sangat Cocok):** {label_counts.get(2, 0)}  
            """
        )

    # ==============================
    # 3.4 RINGKASAN
    # ==============================
    st.markdown("---")
    explanation_card(
        "Ringkasan Menu 3",
        """
        Pada halaman ini telah dilakukan beberapa langkah penting:

        1. **Imputasi missing value** menggunakan **KNN Imputer (k = 3 tetangga)** 
           pada seluruh fitur numerik.  
        2. **Pelabelan rule-based** menggunakan tiga variabel inti: pH, suhu, dan kelembaban,
           dengan output 3 kelas (`0`, `1`, `2`).  
        3. **Analisis distribusi label** untuk memahami komposisi kelas dalam dataset
           sebelum masuk ke tahap balancing (SMOTE) dan pelatihan model.

        Dataset hasil tahap ini disimpan sebagai `df_labeled`  
        dan siap dilanjutkan ke proses berikutnya:  
        pemisahan data, normalisasi, SMOTE, dan pelatihan model.
        """
    )


if __name__ == "__main__":
    main()
