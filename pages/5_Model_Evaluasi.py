import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,   # <--- TAMBAHAN: untuk weighted F1 multiclass
)


# ==============================
# 1. CARD PENJELASAN
# ==============================
def explanation_card(title: str, content: str):
    st.markdown(
        f"""
        <div style="
            background-color:#000000;
            padding:1rem 1.25rem;
            border-left:5px solid #FF9800;
            border-radius:6px;
            margin-bottom:0.5rem;">
            <b>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)


# ==============================
# 2. HELPER: EVALUASI MODEL
# ==============================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Evaluasi model klasifikasi dengan 3 kelas (0, 1, 2).

    Mengembalikan dict berisi:
    - train_acc         : akurasi pada data latih
    - test_acc          : akurasi pada data uji
    - train_f1_weighted : F1-score tertimbang (weighted) pada data latih
    - test_f1_weighted  : F1-score tertimbang (weighted) pada data uji
    - cm                : confusion matrix 3x3 (kelas 0,1,2)
    - report_df         : classification report sebagai DataFrame (per kelas + avg)
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Akurasi
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Weighted F1-score → metrik penting untuk multiclass dengan distribusi tidak seimbang
    train_f1_weighted = f1_score(y_train, y_train_pred, average="weighted")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    # Tetap pakai kelas 0,1,2 (kalau ada kelas yang tidak muncul, nilainya 0 di matriks)
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1, 2],
        target_names=["0 - Tidak Cocok", "1 - Kurang Cocok", "2 - Sangat Cocok"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T

    return {
        "name": name,
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_f1_weighted": train_f1_weighted,
        "test_f1_weighted": test_f1_weighted,
        "cm": cm,
        "report_df": report_df,
        "y_test_pred": y_test_pred,
    }


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ==============================
# 3. MENU UTAMA
# ==============================
def main():
    st.markdown("## 5. Model & Evaluasi")

    # Memastikan hasil dari Menu 4 tersedia
    needed_keys = [
        "X_train_bal",
        "y_train_bal",
        "X_test",
        "y_test",
        "X_train_scaled",
        "X_test_scaled",
    ]
    for key in needed_keys:
        if key not in st.session_state or st.session_state[key] is None:
            st.warning(
                f"Data `{key}` belum tersedia. "
                "Jalankan terlebih dahulu halaman **4. Split Data, SMOTE & Normalisasi**."
            )
            return

    # Mengambil data dari session_state
    X_train_bal = st.session_state["X_train_bal"]
    y_train_bal = st.session_state["y_train_bal"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    X_train_scaled = st.session_state["X_train_scaled"]
    X_test_scaled = st.session_state["X_test_scaled"]

    # Pastikan dalam bentuk array (kalau sebelumnya DataFrame juga tetap aman)
    X_train_model = np.asarray(X_train_scaled)
    X_test_model = np.asarray(X_test_scaled)

    # Coba ambil nama fitur (kalau ada) untuk informasi
    if isinstance(X_train_scaled, pd.DataFrame):
        feature_names = list(X_train_scaled.columns)
    elif isinstance(X_train_bal, pd.DataFrame):
        feature_names = list(X_train_bal.columns)
    else:
        # fallback: kalau tidak ada DataFrame, pakai index generik
        feature_names = [f"F{i}" for i in range(X_train_model.shape[1])]

    # ==============================
    # 5.1 PILIH MODEL & PARAMETER
    # ==============================
    st.markdown("### 5.1 Pemilihan Model dan Pengaturan Hyperparameter")

    explanation_card(
        "Konsep pada halaman ini",
        """
        Halaman ini menggunakan **fitur numerik yang sudah dipilih, diberi label (rule-based), 
        di-*oversampling* (SMOTE multiclass), dan dinormalisasi** pada halaman sebelumnya.

        Tugas model di sini adalah melakukan **klasifikasi 3 kelas**:

        - `0` → Tidak Cocok  
        - `1` → Kurang Cocok  
        - `2` → Sangat Cocok  

        Pada bagian ini dapat dilakukan:

        - pemilihan model yang akan dilatih (Random Forest, MLP, Hybrid Soft Voting, Hybrid Stacking),  
        - pengaturan hyperparameter untuk model yang dipilih,  
        - pelatihan hanya untuk model yang dipilih.  

        Pendekatan ini menjaga beban komputasi tetap terkontrol, 
        sekaligus memudahkan perbandingan hasil antar model dalam konteks **multiclass**.
        """
    )

    st.markdown("**Fitur yang digunakan oleh model:**")
    st.write(feature_names)

    model_choice = st.selectbox(
        "Pilih model yang akan dilatih:",
        [
            "Random Forest",
            "MLP",
            "Hybrid Soft Voting (RF + MLP)",
            "Hybrid Stacking (RF + MLP + LR)",
        ],
    )

    st.markdown("#### 5.1.1 Pengaturan Hyperparameter")

    # Menentukan model mana yang memerlukan pengaturan RF / MLP
    show_rf_params = model_choice in [
        "Random Forest",
        "Hybrid Soft Voting (RF + MLP)",
        "Hybrid Stacking (RF + MLP + LR)",
    ]
    show_mlp_params = model_choice in [
        "MLP",
        "Hybrid Soft Voting (RF + MLP)",
        "Hybrid Stacking (RF + MLP + LR)",
    ]

    # ----- Hyperparameter Random Forest -----
    if show_rf_params:
        with st.expander(
            "Pengaturan Random Forest", expanded=(model_choice == "Random Forest")
        ):
            rf_n_estimators = st.number_input(
                "Jumlah pohon (n_estimators)",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
            )
            rf_max_depth_opt = st.selectbox(
                "Maksimum kedalaman pohon (max_depth)",
                options=["None", 5, 10, 20, 30],
                index=2,  # default 10
            )
            rf_max_depth = None if rf_max_depth_opt == "None" else rf_max_depth_opt

            rf_min_samples_split = st.number_input(
                "min_samples_split",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
            )
    else:
        # Nilai default ketika Random Forest tidak digunakan
        rf_n_estimators = 100
        rf_max_depth = 10
        rf_min_samples_split = 5

    # ----- Hyperparameter MLP -----
    if show_mlp_params:
        with st.expander("Pengaturan MLP", expanded=(model_choice == "MLP")):
            hidden_str = st.text_input(
                "Hidden layer sizes (misal: 32,16)",
                value="32,16",
            )
            try:
                hidden_layer_sizes = tuple(
                    int(x.strip()) for x in hidden_str.split(",") if x.strip() != ""
                )
                if len(hidden_layer_sizes) == 0:
                    hidden_layer_sizes = (32, 16)
            except Exception:
                hidden_layer_sizes = (32, 16)

            mlp_activation = st.selectbox(
                "Activation",
                options=["relu", "tanh", "logistic"],
                index=0,
            )

            mlp_alpha = st.number_input(
                "Alpha (L2 regularization)",
                min_value=1e-6,
                max_value=1.0,
                value=0.0001,
                step=0.0001,
                format="%.6f",
            )

            mlp_max_iter = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
            )

            mlp_early_stopping = st.checkbox(
                "Gunakan early_stopping",
                value=True,
            )

            mlp_validation_fraction = st.number_input(
                "Validation fraction (untuk early_stopping)",
                min_value=0.05,
                max_value=0.4,
                value=0.1,
                step=0.05,
                format="%.2f",
            )
    else:
        # Nilai default ketika MLP tidak digunakan
        hidden_layer_sizes = (32, 16)
        mlp_activation = "relu"
        mlp_alpha = 0.0001
        mlp_max_iter = 500
        mlp_early_stopping = True
        mlp_validation_fraction = 0.1

    # Tombol untuk memulai pelatihan
    train_button = st.button("🚀 Latih Model Terpilih")

    model_trained = False
    selected_result = None

    if train_button:
        # ==============================
        # 5.2 PEMBENTUKAN DAN PELATIHAN MODEL
        # ==============================
        if model_choice == "Random Forest":
            model_name = "Random Forest"
            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=-1,
            )

        elif model_choice == "MLP":
            model_name = "MLP"
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=mlp_activation,
                alpha=mlp_alpha,
                solver="adam",
                max_iter=mlp_max_iter,
                random_state=42,
                early_stopping=mlp_early_stopping,
                validation_fraction=mlp_validation_fraction,
            )

        elif model_choice == "Hybrid Soft Voting (RF + MLP)":
            model_name = "Hybrid Soft Voting"
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=-1,
            )
            mlp_model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=mlp_activation,
                alpha=mlp_alpha,
                solver="adam",
                max_iter=mlp_max_iter,
                random_state=42,
                early_stopping=mlp_early_stopping,
                validation_fraction=mlp_validation_fraction,
            )
            model = VotingClassifier(
                estimators=[("rf", rf_model), ("mlp", mlp_model)],
                voting="soft",
                n_jobs=-1,
            )

        else:  # Hybrid Stacking
            model_name = "Hybrid Stacking"
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=-1,
            )
            mlp_model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=mlp_activation,
                alpha=mlp_alpha,
                solver="adam",
                max_iter=mlp_max_iter,
                random_state=42,
                early_stopping=mlp_early_stopping,
                validation_fraction=mlp_validation_fraction,
            )
            final_est = LogisticRegression(max_iter=500)
            model = StackingClassifier(
                estimators=[("rf", rf_model), ("mlp", mlp_model)],
                final_estimator=final_est,
                n_jobs=-1,
                passthrough=False,
            )

        with st.spinner(f"Melatih {model_name}..."):
            model.fit(X_train_model, y_train_bal)

        # Evaluasi
        selected_result = evaluate_model(
            model_name,
            model,
            X_train_model,
            y_train_bal,
            X_test_model,
            y_test,
        )

        # Menyimpan ke session_state
        st.session_state["last_model_result"] = selected_result
        st.session_state["last_model_object"] = model

        # Opsional: menyimpan berdasarkan tipe model
        if model_name == "Random Forest":
            st.session_state["rf_model"] = model
        elif model_name == "MLP":
            st.session_state["mlp_model"] = model
        elif model_name == "Hybrid Soft Voting":
            st.session_state["voting_model"] = model
        elif model_name == "Hybrid Stacking":
            st.session_state["stacking_model"] = model

        model_trained = True
        st.success(f"Model {model_name} berhasil dilatih dan dievaluasi.")

    # Jika belum ada pelatihan baru, tetapi sudah ada hasil sebelumnya
    if not model_trained and "last_model_result" in st.session_state:
        selected_result = st.session_state["last_model_result"]

    # Jika tidak ada model sama sekali
    if selected_result is None:
        st.info(
            "Belum terdapat model yang dilatih. "
            "Pilih model dan tekan tombol **Latih Model Terpilih** untuk memulai pelatihan."
        )
        return

    # ==============================
    # 5.3 TAMPILAN HASIL MODEL TERPILIH
    # ==============================
    st.markdown("---")
    st.markdown(f"### 5.3 Hasil Evaluasi untuk: **{selected_result['name']}**")

    # 5.3.1 Akurasi + Weighted F1
    st.markdown("#### 5.3.1 Akurasi & Weighted F1 (Train vs Test)")

    explanation_card(
        "Mengapa pakai Weighted F1-score?",
        """
        Pada kasus **3 kelas** dengan distribusi yang mungkin tidak seimbang,
        akurasi saja sering tidak cukup menggambarkan kualitas model.

        - **Accuracy**: seberapa banyak prediksi yang benar secara keseluruhan.  
        - **Weighted F1-score**: rata-rata F1 dari semua kelas, 
          tetapi **ditimbang** berdasarkan jumlah data tiap kelas.  

        Artinya, Weighted F1 memberikan gambaran yang lebih adil ketika ada kelas
        yang datanya jauh lebih sedikit dibanding kelas lainnya.
        """
    )

    acc_df = pd.DataFrame(
        {
            "Jenis": ["Train", "Test"],
            "Accuracy": [selected_result["train_acc"], selected_result["test_acc"]],
            "Weighted F1": [
                selected_result["train_f1_weighted"],
                selected_result["test_f1_weighted"],
            ],
        }
    )

    col_acc_left, col_acc_right = st.columns([1, 1])
    with col_acc_left:
        st.dataframe(
            acc_df.style.format({"Accuracy": "{:.4f}", "Weighted F1": "{:.4f}"})
        )

    with col_acc_right:
        fig_acc, ax_acc = plt.subplots(figsize=(4, 3))
        x = np.arange(len(acc_df["Jenis"]))
        width = 0.35

        ax_acc.bar(x - width / 2, acc_df["Accuracy"], width, label="Accuracy")
        ax_acc.bar(x + width / 2, acc_df["Weighted F1"], width, label="Weighted F1")

        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(acc_df["Jenis"])
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_ylabel("Skor")
        ax_acc.set_title("Perbandingan Accuracy dan Weighted F1\n(Train vs Test)")
        ax_acc.legend()
        st.pyplot(fig_acc, use_container_width=True)

    # 5.3.2 Confusion Matrix
    st.markdown("#### 5.3.2 Confusion Matrix")

    explanation_card(
        "Cara membaca confusion matrix",
        """
        Karena masalah yang dihadapi adalah **klasifikasi 3 kelas** (0, 1, 2),
        confusion matrix yang ditampilkan berukuran **3×3**.

        - Sumbu Y (vertikal) merepresentasikan kelas **aktual**.  
        - Sumbu X (horisontal) merepresentasikan kelas **prediksi**.  
        - Elemen pada diagonal utama menunjukkan prediksi yang benar.  

        Hal-hal yang dapat diperhatikan antara lain:

        - Apakah kelas minoritas (misalnya kelas 2 - Sangat Cocok) sering tertukar dengan kelas lain.  
        - Apakah terdapat kelas yang jarang atau hampir tidak pernah diprediksi oleh model.  
        """
    )

    fig_cm = plot_confusion_matrix(
        selected_result["cm"], f"Confusion Matrix - {selected_result['name']}"
    )
    st.pyplot(fig_cm, use_container_width=True)

    # 5.3.3 Classification Report
    st.markdown("#### 5.3.3 Classification Report")

    explanation_card(
        "Interpretasi classification report (multiclass)",
        """
        Tabel classification report menampilkan metrik utama untuk **setiap kelas** (0, 1, 2)
        serta ringkasan di baris terakhir (macro avg dan weighted avg):

        - **Precision** tinggi → ketika model menebak kelas tersebut, jarang salah.  
        - **Recall** tinggi → sebagian besar data kelas tersebut berhasil dikenali.  
        - **F1-score** → rata-rata harmonis antara precision dan recall.  

        Baris **`weighted avg`** merangkum performa keseluruhan model dengan
        mempertimbangkan **proporsi data tiap kelas**.  
        Ini sangat berguna ketika jumlah data di kelas 0, 1, dan 2 tidak seimbang.
        """
    )

    st.dataframe(selected_result["report_df"].style.format("{:.4f}"))

    # ==============================
    # 5.4 RINGKASAN
    # ==============================
    st.markdown("---")
    explanation_card(
        "Ringkasan Menu 5",
        f"""
        Untuk model **{selected_result['name']}**, pada Menu 5 dilakukan beberapa langkah utama:

        1. **Pemilihan model dan pengaturan hyperparameter**  
           Model dilatih untuk tugas klasifikasi **3 kelas** (`0`, `1`, `2`)
           menggunakan fitur yang sudah di-*preprocess* pada Menu 3–4.  

        2. **Pelatihan model**  
           Menggunakan data latih hasil SMOTE multiclass (jika diterapkan) 
           dan fitur yang sudah dinormalisasi.  

        3. **Evaluasi model (multiclass)**  
           - Akurasi pada data latih dan data uji,  
           - **Weighted F1-score** sebagai metrik utama yang lebih adil pada data tidak seimbang,  
           - Confusion matrix 3×3,  
           - Classification report lengkap per kelas (0, 1, 2) beserta ringkasan `weighted avg`.  

        Model terakhir yang dilatih disimpan di `session_state['last_model_object']`
        dan dapat dimanfaatkan pada halaman berikutnya, misalnya untuk analisis interpretabilitas
        maupun prediksi interaktif.
        """
    )


if __name__ == "__main__":
    main()
