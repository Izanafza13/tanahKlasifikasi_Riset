import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. CARD PENJELASAN
# ==============================
def explanation_card(title: str, content: str):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #03A9F4;
            border-radius:6px;
            margin-bottom:0.5rem;">
            <b>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)


# ==============================
# 2. ATURAN RULE-BASED ADAPTIF
#    (SAMA DENGAN MENU 3: 3 FITUR)
# ==============================
def label_kesesuaian_lahan(row):
    """
    Rule-based kesesuaian lahan padi sawah.

    Fitur opsional yang dipakai:
      - pH_tanah
      - suhu_tanah
      - kelembaban_udara

    Kalau suatu fitur tidak ada di input → di-skip.

    Output:
      0 = Tidak Cocok
      1 = Kurang Cocok
      2 = Sangat Cocok
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

    # -------------------- Suhu tanah --------------------
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

    # -------------------- Kelembaban udara --------------------
    if "kelembaban_udara" in row.index:
        kelembaban = row["kelembaban_udara"]
        if kelembaban < 30:
            status_RH = 0
        elif 33 <= kelembaban <= 90:
            status_RH = 2
        else:
            status_RH = 1
        statuses.append(status_RH)

    # -------------------- Penentuan label akhir --------------------
    if len(statuses) == 0:
        # kalau tidak ada fitur sama sekali → netral → 1 (Kurang Cocok)
        return 1

    # Hukum faktor pembatas (minimum)
    if 0 in statuses:
        return 0
    elif all(s == 2 for s in statuses):
        return 2
    else:
        return 1


label_map = {
    0: "0 - Tidak Cocok",
    1: "1 - Kurang Cocok",
    2: "2 - Sangat Cocok",
}


# ==============================
# 3. MENU UTAMA
# ==============================
def main():
    st.markdown("## 7. Prediksi Interaktif Kesesuaian Lahan")

    explanation_card(
        "Tujuan halaman ini",
        """
        Halaman ini digunakan untuk:

        - Mengisi **semua fitur yang dipakai model** (mengikuti pilihan fitur di Menu 1 & proses di Menu 4).  
        - Memilih **model yang sudah dilatih** di Menu 5.  
        - Melihat:
          - prediksi kelas kesesuaian lahan dari model (0, 1, atau 2),  
          - probabilitas tiap kelas (jika model mendukung `predict_proba`),  
          - perbandingan dengan **label rule-based adaptif** yang menggunakan
            `pH_tanah`, `suhu_tanah`, dan `kelembaban_udara` (sama dengan Menu 3).  
        """
    )

    # ---------- CEK OBJEK WAJIB ----------
    if "scaler" not in st.session_state or st.session_state["scaler"] is None:
        st.warning(
            "Objek scaler tidak ditemukan. "
            "Jalankan terlebih dahulu halaman **4. Split Data, Normalisasi & SMOTE**."
        )
        return

    if "feature_cols" not in st.session_state or not st.session_state["feature_cols"]:
        st.warning(
            "Informasi fitur (`feature_cols`) tidak ditemukan. "
            "Pastikan Menu 4 sudah dijalankan tanpa error."
        )
        return

    scaler = st.session_state["scaler"]
    feature_cols = st.session_state["feature_cols"]

    # data latih untuk referensi nilai default input (boleh scaled)
    X_source = None
    if "X_train_bal" in st.session_state and st.session_state["X_train_bal"] is not None:
        X_source = st.session_state["X_train_bal"]
    elif "X_train_scaled" in st.session_state and st.session_state["X_train_scaled"] is not None:
        X_source = st.session_state["X_train_scaled"]

    if X_source is None or not isinstance(X_source, (pd.DataFrame,)):
        st.warning(
            "Data latih untuk referensi fitur tidak tersedia. "
            "Pastikan Menu 4 sudah selesai dijalankan."
        )
        return

    # ---------- PILIH MODEL ----------
    available_models = {}
    labels = []

    if "last_model_object" in st.session_state and "last_model_result" in st.session_state:
        last_name = st.session_state["last_model_result"]["name"]
        labels.append(f"Model terakhir: {last_name}")
        available_models[f"Model terakhir: {last_name}"] = st.session_state["last_model_object"]

    if "rf_model" in st.session_state:
        labels.append("Random Forest (terakhir dilatih)")
        available_models["Random Forest (terakhir dilatih)"] = st.session_state["rf_model"]

    if "mlp_model" in st.session_state:
        labels.append("MLP (terakhir dilatih)")
        available_models["MLP (terakhir dilatih)"] = st.session_state["mlp_model"]

    if "voting_model" in st.session_state:
        labels.append("Hybrid Soft Voting")
        available_models["Hybrid Soft Voting"] = st.session_state["voting_model"]

    if "stacking_model" in st.session_state:
        labels.append("Hybrid Stacking")
        available_models["Hybrid Stacking"] = st.session_state["stacking_model"]

    if not labels:
        st.warning(
            "Belum terdapat model yang siap digunakan untuk prediksi. "
            "Latih minimal satu model pada **5. Model & Evaluasi**."
        )
        return

    model_label = st.selectbox(
        "Pilih model yang akan digunakan untuk prediksi:",
        labels,
    )
    model = available_models[model_label]

    st.markdown(
        f"**Model aktif:** `{model_label}` "
        f"(tipe objek: `{type(model).__name__}`)"
    )

    if hasattr(model, "classes_"):
        model_classes = list(model.classes_)
    else:
        model_classes = [0, 1, 2]

    # ---------- INPUT SEMUA FITUR ----------
    st.markdown("---")
    st.markdown("### 7.1 Input Kondisi Lahan (semua fitur yang dipakai model)")

    input_values = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_cols):
        with cols[i % 3]:
            if col in X_source.columns:
                default_val = float(X_source[col].mean())
            else:
                default_val = 0.0

            if col == "pH_tanah":
                val = st.number_input(
                    "pH Tanah (pH_tanah)",
                    min_value=3.0,
                    max_value=9.0,
                    value=6.5,
                    step=0.1,
                )
            elif col == "suhu_tanah":
                val = st.number_input(
                    "Suhu Tanah (°C) [suhu_tanah]",
                    min_value=15.0,
                    max_value=45.0,
                    value=29.0,
                    step=0.5,
                )
            elif col == "kelembaban_udara":
                val = st.number_input(
                    "Kelembaban Udara (%) [kelembaban_udara]",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                )
            else:
                val = st.number_input(col, value=float(default_val))

            input_values[col] = val

    input_model_df = pd.DataFrame([input_values], columns=feature_cols)

    st.markdown("**Input yang akan masuk ke model (semua fitur):**")
    st.dataframe(input_model_df)

    # ---------- TOMBOL PREDIKSI ----------
    predict_btn = st.button("🔮 Jalankan Prediksi")

    if not predict_btn:
        st.info(
            "Isi nilai semua fitur, pilih model, lalu tekan **Jalankan Prediksi** "
            "untuk melihat hasil."
        )
        return

    # ---------- PROSES PREDIKSI ----------
    try:
        X_scaled = scaler.transform(input_model_df[feature_cols])
    except Exception as e:
        st.error(f"Gagal melakukan scaling pada input: {e}")
        return

    try:
        y_pred = model.predict(X_scaled)[0]
    except Exception as e:
        st.error(f"Gagal melakukan prediksi menggunakan model: {e}")
        return

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            proba_vec = model.predict_proba(X_scaled)[0]
            probs = dict(zip(model_classes, proba_vec))
        except Exception:
            probs = None

    # Rule-based adaptif (3 fitur, sama dengan Menu 3)
    rule_label = label_kesesuaian_lahan(input_model_df.iloc[0])

    # ---------- TAMPILKAN HASIL ----------
    st.markdown("---")
    st.markdown("### 7.2 Hasil Prediksi")

    col_out_left, col_out_right = st.columns([1, 1])

    with col_out_left:
        st.markdown("**Hasil model pembelajaran mesin:**")
        st.markdown(f"- Prediksi model: **{label_map.get(int(y_pred), y_pred)}**")

        st.markdown("**Hasil aturan rule-based (3 fitur):**")
        st.markdown(f"- Label rule-based: **{label_map.get(int(rule_label), rule_label)}**")

        if int(y_pred) == int(rule_label):
            st.markdown(
                "Model dan rule-based memberikan **label yang sama** untuk kondisi input ini."
            )
        else:
            st.markdown(
                "Model dan rule-based memberikan **label yang berbeda** untuk kondisi input ini."
            )

    with col_out_right:
        if probs is not None:
            prob_rows = []
            for c, p in probs.items():
                prob_rows.append(
                    {
                        "kelas": label_map.get(int(c), str(c)),
                        "probabilitas": p,
                    }
                )
            prob_df = pd.DataFrame(prob_rows)

            st.markdown("**Probabilitas prediksi model per kelas:**")
            st.dataframe(prob_df.style.format({"probabilitas": "{:.4f}"}))

            fig_prob, ax_prob = plt.subplots(figsize=(4, 3))
            ax_prob.bar(prob_df["kelas"], prob_df["probabilitas"])
            ax_prob.set_ylim(0, 1.05)
            ax_prob.set_ylabel("Probabilitas")
            ax_prob.set_title("Distribusi Probabilitas Prediksi")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig_prob, use_container_width=True)
        else:
            st.info(
                "Model ini tidak menyediakan metode `predict_proba`, "
                "sehingga probabilitas per kelas tidak dapat ditampilkan."
            )


if __name__ == "__main__":
    main()
