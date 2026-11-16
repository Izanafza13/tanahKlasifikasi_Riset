import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


# ==============================
# 1. CARD PENJELASAN
# ==============================
def explanation_card(title: str, content: str):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #9C27B0;
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
# 2. MENU UTAMA
# ==============================
def main():

    st.markdown("## 4. Split Data, Normalisasi & SMOTE")

    # Cek apakah sudah ada df_labeled
    if "df_labeled" not in st.session_state or st.session_state["df_labeled"] is None:
        st.warning(
            "Dataset belum memiliki label. "
            "Silakan selesaikan terlebih dahulu halaman **3. Distribusi Label & Imputasi**."
        )
        return

    df = st.session_state["df_labeled"].copy()

    if "label_lahan" not in df.columns:
        st.error("Kolom label_lahan tidak ditemukan. Pastikan Page 3 sudah selesai.")
        return

    # Fitur & label
    X = df.drop(columns=["label_lahan"])
    y = df["label_lahan"]   # berisi 3 kelas: 0, 1, 2 (multiclass dari Page 3)

    feature_cols = list(X.columns)       # *** penting untuk Page 7
    st.session_state["feature_cols"] = feature_cols

    # ==============================
    # 4.1 TRAIN–TEST SPLIT
    # ==============================
    st.markdown("### 4.1 Pemisahan Train–Test")

    explanation_card(
        "Tujuan pemisahan data",
        """
        Dataset dibagi menjadi bagian *latih* dan *uji* agar model dapat
        dievaluasi secara objektif pada data yang belum pernah dilihat sebelumnya.

        Dengan cara ini, kita bisa melihat apakah model benar-benar memahami pola
        pada data, bukan sekadar "menghapal" data latih saja.
        """
    )

    test_size = st.slider(
        "Proporsi data untuk test",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    st.success(f"Data berhasil di-split: {len(X_train)} train, {len(X_test)} test.")

    # simpan X_test asli (dipakai di interpretabilitas)
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    # ==============================
    # 4.2 SMOTE (dilakukan sebelum scaling)
    # ==============================
    st.markdown("---")
    st.markdown("### 4.2 SMOTE (Multiclass, Opsional)")

    explanation_card(
        "Kapan SMOTE (multiclass) diperlukan?",
        """
        Pada halaman 3, setiap baris data sudah diberi label `label_lahan` 
        dengan **3 kelas**:

        - `0` → Tidak Cocok  
        - `1` → Kurang Cocok  
        - `2` → Sangat Cocok  

        Sering kali, jumlah data untuk tiap kelas **tidak seimbang** 
        (misalnya kelas `1` jauh lebih banyak daripada kelas `0` dan `2`).  
        Jika dibiarkan, model bisa menjadi **bias**, cenderung menebak ke kelas yang paling banyak.

        Di sinilah **SMOTE (Synthetic Minority Over-sampling Technique)** digunakan:

        - SMOTE membuat **sampel sintetis baru** untuk kelas yang jumlahnya sedikit,
          bukan sekadar menggandakan baris lama.
        - Teknik ini bekerja untuk kasus **multiclass**, sehingga semua kelas 
          (`0`, `1`, dan `2`) bisa dibuat lebih seimbang pada data latih.
        """
    )

    apply_smote = st.checkbox("Gunakan SMOTE untuk menyeimbangkan kelas (multiclass)", value=True)

    if apply_smote:
        # SMOTE multiclass: langsung bekerja dengan label 0/1/2
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        st.success("SMOTE (multiclass) berhasil diterapkan pada data latih.")
        st.write("Distribusi label setelah SMOTE (train):")
        st.write(y_train_bal.value_counts().sort_index())
    else:
        st.info("SMOTE tidak diterapkan. Menggunakan data latih asli apa adanya.")
        X_train_bal = X_train
        y_train_bal = y_train

    # Simpan untuk Page 5
    st.session_state["X_train_bal"] = X_train_bal
    st.session_state["y_train_bal"] = y_train_bal

    # ==============================
    # 4.3 NORMALISASI (SETELAH SMOTE)
    # ==============================
    st.markdown("---")
    st.markdown("### 4.3 Normalisasi Min-Max")

    explanation_card(
        "Mengapa normalisasi dilakukan setelah SMOTE?",
        """
        Prosesnya dibuat berurutan agar hasilnya lebih masuk akal secara statistik:

        1. **SMOTE dulu pada skala asli fitur**  
           → supaya interpolasi titik baru (sampel sintetis) terjadi pada nilai asli
             suhu, pH, dan kelembaban.

        2. **Setelah itu baru dinormalisasi dengan Min-Max**  
           → semua fitur dibawa ke rentang 0–1 sehingga:
             - model yang peka terhadap skala (seperti MLP) lebih stabil,
             - tidak ada satu fitur yang dominasinya terlalu besar hanya karena skala angkanya lebih besar.
        """
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train_bal)

    X_train_scaled = scaler.transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    df_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    df_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    st.markdown("**Contoh hasil normalisasi (5 baris pertama data train):**")
    st.dataframe(df_train_scaled.head())

    # Simpan ke session_state
    st.session_state["scaler"] = scaler
    st.session_state["X_train_scaled"] = df_train_scaled
    st.session_state["X_test_scaled"] = df_test_scaled
    st.session_state["y_train"] = y_train_bal  # *** penting untuk page 5
    st.session_state["y_test"] = y_test

    # ==============================
    # 4.4 RINGKASAN
    # ==============================
    st.markdown("---")
    explanation_card(
        "Ringkasan Menu 4",
        f"""
        Pada menu ini dilakukan:

        1. **Split train–test** (test size = {test_size}).  
        2. **SMOTE multiclass (opsional)** pada data latih untuk menyeimbangkan
           ketiga kelas label (`0`, `1`, `2`).  
        3. **Normalisasi Min-Max** untuk seluruh fitur setelah SMOTE.  
        4. Penyimpanan variabel penting untuk Menu 5–7:
           - `X_train_bal`, `y_train_bal`  
           - `X_train_scaled`, `X_test_scaled`  
           - `X_test`, `y_test`  
           - `scaler`  
           - `feature_cols`
        """
    )


if __name__ == "__main__":
    main()
