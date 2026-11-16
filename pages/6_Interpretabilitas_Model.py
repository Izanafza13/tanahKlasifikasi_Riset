import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ==============================
# 1. CARD PENJELASAN
# ==============================
def explanation_card(title: str, content: str):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #4CAF50;
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
    st.markdown("## 6. Interpretabilitas Model")

    # ---------------------------------
    # 2.1 Pastikan model & data tersedia
    # ---------------------------------
    if "last_model_object" not in st.session_state or "last_model_result" not in st.session_state:
        st.warning(
            "Belum ada model yang dilatih.\n"
            "Silakan latih model terlebih dahulu di halaman **5. Model & Evaluasi**."
        )
        return

    model = st.session_state["last_model_object"]
    res = st.session_state["last_model_result"]

    needed_keys = [
        "X_train_bal", "y_train_bal",
        "X_test", "y_test",
        "X_train_scaled", "X_test_scaled",
    ]
    for key in needed_keys:
        if key not in st.session_state or st.session_state[key] is None:
            st.warning(
                f"Data `{key}` belum tersedia.\n"
                "Pastikan halaman **4. Split Data, SMOTE & Normalisasi** sudah dijalankan."
            )
            return

    # Data dari Page 4
    X_train_bal = st.session_state["X_train_bal"]
    y_train = st.session_state["y_train_bal"]
    X_test = st.session_state["X_test"]              # DataFrame (untuk ditampilkan)
    y_test = st.session_state["y_test"]
    X_train_scaled = st.session_state["X_train_scaled"]  # dipakai model
    X_test_scaled = st.session_state["X_test_scaled"]    # dipakai model

    # Pastikan bentuk ke model = numpy array
    if isinstance(X_train_scaled, pd.DataFrame):
        X_train_model = X_train_scaled.to_numpy()
    else:
        X_train_model = X_train_scaled

    if isinstance(X_test_scaled, pd.DataFrame):
        X_test_model = X_test_scaled.to_numpy()
    else:
        X_test_model = X_test_scaled

    # Nama fitur pakai kolom X_test (DataFrame asli, belum scale)
    feature_cols = list(X_test.columns)

    # Info kelas model (dinamis)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = [0, 1, 2]

    class_name_map = {
        0: "0 - Tidak Cocok",
        1: "1 - Kurang Cocok",
        2: "2 - Sangat Cocok",
    }
    class_names = [class_name_map.get(int(c), str(c)) for c in classes]

    # Tipe model (untuk pemilihan metode feature importance)
    is_rf = isinstance(model, RandomForestClassifier)
    is_mlp = isinstance(model, MLPClassifier)

    # ==============================
    # 6.1 Informasi model
    # ==============================
    st.markdown("### 6.1 Informasi Model")

    explanation_card(
        "Model yang sedang dijelaskan",
        """
        Bagian ini menjelaskan **model terakhir yang dilatih di Menu 5**. 
        Informasi yang ditampilkan antara lain:

        - jenis model dan tingkat akurasi pada data latih dan data uji,  
        - (jika tersedia) ringkasan metrik lain seperti **Weighted F1-score** untuk kasus 3 kelas,  
        - pentingnya masing-masing fitur secara global,  
        - penjelasan satu contoh data uji lengkap dengan probabilitas prediksi per kelas (0, 1, 2).  
        """
    )

    col_info_left, col_info_right = st.columns(2)
    with col_info_left:
        st.write("**Nama model:**", res["name"])
        st.write("**Tipe objek:**", type(model).__name__)
    with col_info_right:
        st.write("**Akurasi Train:**", f"{res['train_acc']:.4f}")
        st.write("**Akurasi Test:**", f"{res['test_acc']:.4f}")
        # Tambahan: tampilkan Weighted F1 kalau ada (sinkron dengan Page 5)
        train_f1 = res.get("train_f1_weighted", None)
        test_f1 = res.get("test_f1_weighted", None)
        if train_f1 is not None and test_f1 is not None:
            st.write("**Weighted F1 Train:**", f"{train_f1:.4f}")
            st.write("**Weighted F1 Test:**", f"{test_f1:.4f}")

    # ==============================
    # 6.2 Pentingnya fitur (global)
    # ==============================
    st.markdown("---")
    st.markdown("### 6.2 Pentingnya Fitur (Global Feature Importance)")

    explanation_card(
        "Apa yang dimaksud pentingnya fitur?",
        """
        Di sini ditunjukkan seberapa besar peran masing-masing fitur ketika model
        membedakan kelas kesesuaian lahan (0, 1, dan 2).

        Ringkasnya:

        - Fitur dengan nilai **importance** lebih tinggi dianggap lebih sering dipakai model
          untuk mengambil keputusan.  
        - Untuk **Random Forest**, nilai importance diambil langsung dari atribut 
          `feature_importances_`.  
        - Untuk model lain (misalnya MLP, Voting, Stacking), dipakai
          **Permutation Importance**, yaitu seberapa besar penurunan performa model
          ketika nilai satu fitur diacak.  

        Jadi grafik ini membantu menjawab pertanyaan:
        “Fitur mana yang paling didengar oleh model ketika menentukan kelas lahan?”.
        """
    )

    # Hitung importance
    if is_rf and hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    else:
        perm = permutation_importance(
            model,
            X_test_model,
            y_test,
            n_repeats=15,
            random_state=42,
            n_jobs=-1,
        )
        importances = perm.importances_mean

    # Safety: panjang vector importance vs jumlah fitur
    if len(importances) != len(feature_cols):
        st.warning(
            f"Panjang vektor importance ({len(importances)}) tidak sama dengan jumlah fitur ({len(feature_cols)}). "
            "Nama fitur akan disesuaikan dengan panjang yang paling pendek."
        )
        min_len = min(len(importances), len(feature_cols))
        importances = importances[:min_len]
        feature_cols_effective = feature_cols[:min_len]
    else:
        feature_cols_effective = feature_cols

    # Urutkan dari yang paling penting
    idx = np.argsort(importances)[::-1]
    sorted_feat = [feature_cols_effective[i] for i in idx]
    sorted_imp = importances[idx]

    col_fi_left, col_fi_right = st.columns(2)

    with col_fi_left:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(sorted_feat[::-1], sorted_imp[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Global Feature Importance")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_fi_right:
        fi_df = pd.DataFrame({"Fitur": sorted_feat, "Importance": sorted_imp})
        st.markdown("**Tabel pentingnya fitur:**")
        st.dataframe(fi_df.style.format({"Importance": "{:.4f}"}))

        st.markdown(
            """
            **Catatan singkat:**

            - Fitur di baris paling atas memiliki pengaruh paling besar menurut model.  
            - Nilai *importance* yang lebih besar menunjukkan kontribusi yang lebih kuat.  
            - Jika satu fitur memiliki importance jauh di atas yang lain,
              artinya model sangat mengandalkan fitur tersebut untuk membedakan 0, 1, dan 2.
            """
        )

    # ==============================
    # 6.3 Analisis lokal (satu sampel)
    # ==============================
    st.markdown("---")
    st.markdown("### 6.3 Analisis Lokal: Satu Sampel Uji")

    explanation_card(
        "Apa yang dilihat di analisis lokal?",
        """
        Pada bagian ini, fokus diarahkan ke **satu baris data uji**. 
        Untuk sampel tersebut ditampilkan:

        - nilai setiap fitur (dalam skala aslinya),  
        - label aktual dari data (0, 1, atau 2),  
        - prediksi model (0, 1, atau 2),  
        - jika tersedia, **probabilitas** model untuk masing-masing kelas (0, 1, 2).  

        Dengan tampilan ini, pengguna awam bisa diajak memahami:

        > “Untuk kondisi lahan seperti ini, model memilih kelas X
        > dengan keyakinan sekian persen, dan tetap mempertimbangkan dua kelas lainnya.”
        """
    )

    if len(X_test) == 0:
        st.warning("Data uji kosong. Tidak bisa melakukan analisis lokal.")
        return

    idx_sample = st.slider(
        "Pilih indeks sampel dari data uji:",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    # Data asli (belum di-scale) untuk ditampilkan
    x_raw = X_test.iloc[idx_sample]

    # Data yang masuk ke model (sudah di-scale)
    if isinstance(X_test_scaled, pd.DataFrame):
        x_scaled = X_test_scaled.iloc[idx_sample].to_numpy().reshape(1, -1)
    else:
        x_scaled = X_test_model[idx_sample].reshape(1, -1)

    y_true = y_test.iloc[idx_sample]
    y_pred = model.predict(x_scaled)[0]

    col_loc_left, col_loc_right = st.columns(2)

    # --- kiri: nilai fitur & label ---
    with col_loc_left:
        st.markdown("**Nilai fitur (skala asli):**")
        st.dataframe(x_raw.to_frame(name="nilai").T)

        st.markdown(
            f"""
            **Kelas aktual (y_true):** `{int(y_true)}`  
            **Kelas prediksi model:** `{int(y_pred)}`  
            """
        )

    # --- kanan: probabilitas per kelas ---
    with col_loc_right:
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x_scaled)[0]  # panjang = len(classes)
                probs = dict(zip(classes, proba))
            except Exception:
                probs = None

        if probs is not None:
            rows = []
            for c, p in probs.items():
                rows.append(
                    {
                        "Kelas": class_name_map.get(int(c), str(c)),
                        "Probabilitas": p,
                    }
                )
            prob_df = pd.DataFrame(rows)

            st.markdown("**Probabilitas prediksi per kelas:**")
            st.dataframe(prob_df.style.format({"Probabilitas": "{:.4f}"}))

            fig_prob, ax_prob = plt.subplots(figsize=(4, 3))
            ax_prob.bar(prob_df["Kelas"], prob_df["Probabilitas"])
            ax_prob.set_ylim(0, 1.05)
            ax_prob.set_ylabel("Probabilitas")
            ax_prob.set_title("Distribusi Probabilitas Prediksi")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig_prob, use_container_width=True)
        else:
            st.info(
                "Model tidak menyediakan `predict_proba`, sehingga probabilitas per kelas "
                "tidak dapat ditampilkan."
            )

    # ==============================
    # 6.4 Ringkasan
    # ==============================
    st.markdown("---")
    explanation_card(
        "Ringkasan Menu 6",
        f"""
        Untuk model **{res['name']}**, di Menu 6 diperoleh dua sudut pandang utama:

        1. **Pentingnya fitur secara global**  
           → menunjukkan fitur mana yang paling banyak digunakan model untuk
             membedakan kelas 0 (Tidak Cocok), 1 (Kurang Cocok), dan 2 (Sangat Cocok).

        2. **Analisis lokal untuk satu sampel uji**  
           → menampilkan nilai fitur, label aktual, prediksi model, dan probabilitas per kelas (jika tersedia).  

        Kombinasi keduanya dapat digunakan sebagai bahan penjelasan dalam laporan:
        - secara global: “fitur apa yang paling berpengaruh?”,  
        - secara lokal: “pada contoh lahan tertentu, model memilih kelas apa dan seberapa yakin?”.  
        """
    )


if __name__ == "__main__":
    main()
