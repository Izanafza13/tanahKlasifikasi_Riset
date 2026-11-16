import streamlit as st
import pandas as pd


# ==========================
# 0. DETEKSI KOLOM WAKTU
# ==========================
def is_temporal_column(col_name: str) -> bool:
    """
    Mengidentifikasi kolom yang kemungkinan berisi informasi waktu
    (tahun, tanggal, bulan, jam, dsb.) berdasarkan nama kolom.
    """
    norm = col_name.strip().lower().replace(" ", "_")
    keywords = [
        "tahun", "thn", "year", "yr",
        "tanggal", "tgl", "date", "datetime",
        "time", "waktu",
        "bulan", "month",
    ]
    return any(k in norm for k in keywords)


# ==========================
# 1. VALIDASI DATA
# ==========================
def validate_raw_data(df: pd.DataFrame, selected_features):
    """
    Validasi dasar untuk data mentah berdasarkan fitur yang dipilih user.
    Mengembalikan:
      - errors: daftar pesan error (bersifat menghentikan proses)
      - warnings: daftar pesan peringatan (masih bisa diproses)
    """
    errors = []
    warnings = []

    # 1. Pastikan semua fitur yang dipilih ada di dataset
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Kolom yang dipilih tidak ditemukan di dataset: {missing_cols}. "
            "Periksa kembali nama kolom atau pilihan fitur."
        )

    # 2. Pastikan semuanya numerik
    numeric_problem = []
    for col in selected_features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            numeric_problem.append(col)

    if numeric_problem:
        errors.append(
            f"Kolom berikut tidak bertipe numerik: {numeric_problem}. "
            "Fitur yang dipilih harus berisi nilai angka."
        )

    if errors:
        return errors, warnings

    # 3. Cek sedikit rentang wajar (opsional, hanya warning)
    if "temperature" in selected_features and "temperature" in df.columns:
        if (df["temperature"] < -10).any() or (df["temperature"] > 60).any():
            warnings.append(
                "Nilai 'temperature' ada yang di luar rentang wajar -10–60 °C."
            )

    if "humidity" in selected_features and "humidity" in df.columns:
        if (df["humidity"] < 0).any() or (df["humidity"] > 100).any():
            warnings.append(
                "Nilai 'humidity' ada yang di luar rentang wajar 0–100%."
            )

    if "ph" in selected_features and "ph" in df.columns:
        if (df["ph"] < 0).any() or (df["ph"] > 14).any():
            warnings.append(
                "Nilai 'ph' ada yang di luar rentang wajar 0–14."
            )

    # Warning sederhana untuk fitur lain: nilai negatif
    for col in selected_features:
        if col in df.columns and (df[col] < 0).any():
            warnings.append(
                f"Ada nilai negatif pada kolom '{col}'. Cek lagi apakah itu masuk akal."
            )

    # 4. Missing value hanya pada fitur yang dipilih
    total_missing = int(df[selected_features].isnull().sum().sum())
    if total_missing > 0:
        warnings.append(
            f"Terdapat {total_missing} missing value pada fitur yang dipilih. "
            "Perlu dipertimbangkan imputasi pada tahap berikutnya."
        )

    return errors, warnings


# ==========================
# 2. MAIN
# ==========================
def main():
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
        st.session_state["file_name"] = None
        st.session_state["selected_features"] = []
        st.session_state["df_raw_with_label"] = None

    st.markdown("## 1. Pendahuluan & Pemuatan Data")

    st.write(
        """
        Pada halaman ini Anda dapat:

        1. Mengunggah file CSV yang berisi data pengukuran lahan.
        2. Melihat ringkasan awal struktur data.
        3. Memilih kolom numerik yang akan digunakan sebagai fitur (minimal 3 kolom).

        Kolom yang berisi informasi waktu seperti tahun atau tanggal akan diabaikan
        dan tidak digunakan sebagai fitur.
        """
    )

    st.markdown("---")
    st.markdown("### Upload Dataset CSV")

    uploaded_file = st.file_uploader(
        "Unggah file CSV (misalnya data sensor atau data lahan):",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df_raw_orig = pd.read_csv(uploaded_file)

            # Simpan versi asli (misalnya bila nanti ingin melihat kembali dengan label)
            st.session_state["df_raw_with_label"] = df_raw_orig.copy()

            # Jika ada kolom label, keluarkan dari data fitur
            if "label" in df_raw_orig.columns:
                df_raw = df_raw_orig.drop(columns=["label"])
            else:
                df_raw = df_raw_orig.copy()

            # Drop kolom yang diduga berisi informasi waktu (tahun, tanggal, dsb.)
            temporal_cols = [col for col in df_raw.columns if is_temporal_column(col)]
            if temporal_cols:
                df_raw = df_raw.drop(columns=temporal_cols)
                st.info(
                    "Kolom yang diduga berisi informasi waktu tidak diikutkan dalam pemrosesan fitur: "
                    f"{temporal_cols}"
                )

            st.session_state["df_raw"] = df_raw
            st.session_state["file_name"] = uploaded_file.name

            st.success(f"Berkas **{uploaded_file.name}** berhasil dibaca.")
        except Exception as e:
            st.error(f"Gagal membaca berkas CSV: {e}")

    df_raw = st.session_state.get("df_raw", None)
    file_name = st.session_state.get("file_name", None)

    if df_raw is None:
        st.info("Silakan upload file CSV terlebih dahulu.")
        return

    # ==========================
    # Tinjauan awal dataset
    # ==========================
    st.markdown("---")
    st.markdown("### Tinjauan Awal Dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Nama berkas:**", file_name or "(tanpa nama)")
        st.write(
            "**Dimensi data (setelah pembuangan label dan kolom waktu, jika ada):**",
            f"{df_raw.shape[0]} baris × {df_raw.shape[1]} kolom",
        )
    with col2:
        st.write("**Nama kolom yang digunakan:**", list(df_raw.columns))
        st.write(
            "**Total missing value (semua kolom):**",
            int(df_raw.isnull().sum().sum()),
        )

    st.write("**Lima baris pertama dataset:**")
    st.dataframe(df_raw.head())

    # ==========================
    # Pemilihan fitur
    # ==========================
    st.markdown("---")
    st.markdown("### Pilih Fitur dari Kolom Dataset")

    st.write(
        """
        Pilih kolom numerik yang akan digunakan sebagai fitur.
        Minimal tiga kolom perlu dipilih agar analisis dapat dilanjutkan.
        Hanya kolom yang dipilih di sini yang akan digunakan pada tahap berikutnya.
        """
    )

    # Hanya tawarkan kolom numerik sebagai kandidat fitur
    numeric_cols = [
        col for col in df_raw.columns
        if pd.api.types.is_numeric_dtype(df_raw[col])
    ]

    if not numeric_cols:
        st.error(
            "Tidak ada kolom numerik yang ditemukan di dataset setelah pembuangan kolom waktu. "
            "Periksa kembali file CSV."
        )
        return

    selected_features = []

    cols = st.columns(3)
    for i, col_name in enumerate(numeric_cols):
        with cols[i % 3]:
            checked = st.checkbox(col_name, value=True, key=f"feat_{col_name}")
            if checked:
                selected_features.append(col_name)

    if len(selected_features) < 3:
        st.warning("Pilih minimal **3 fitur** untuk dapat melanjutkan.")
        st.session_state["selected_features"] = selected_features
        return

    st.session_state["selected_features"] = selected_features
    st.success(f"Fitur yang dipilih: {selected_features}")

    # ==========================
    # Validasi struktur
    # ==========================
    st.markdown("---")
    st.markdown("### Validasi Berdasarkan Fitur yang Dipilih")

    errors, warnings_list = validate_raw_data(df_raw, selected_features)

    if errors:
        st.error("Ditemukan masalah pada struktur dataset:")
        for msg in errors:
            st.write(f"- {msg}")
        st.warning(
            "Perbaiki masalah di atas (misalnya tipe data atau nama kolom) "
            "sebelum melanjutkan ke halaman berikutnya."
        )
        return
    else:
        st.success("Struktur dataset sudah sesuai dengan fitur yang dipilih.")

    if warnings_list:
        st.warning("Beberapa peringatan terdeteksi:")
        for msg in warnings_list:
            st.write(f"- {msg}")
    else:
        st.info("Tidak ada peringatan tambahan untuk fitur yang dipilih.")

    st.markdown(
        """
        Setelah langkah ini selesai, Anda dapat melanjutkan ke halaman:

        - **Deskriptif Data** untuk melihat statistik dan distribusi fitur, atau  
        - Langsung ke tahapan pemrosesan berikutnya.
        """
    )


if __name__ == "__main__":
    main()
