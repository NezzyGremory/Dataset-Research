from __future__ import annotations


METHOD_INFO = {
    # ============================================================
    # CLASSIFICATION
    # ============================================================

    "logistic_regression": {
        "name": "Logistic Regression",
        "category": "Classification",
        "suitable_tasks": [
            "binary_classification",
            "multiclass_classification",
        ],
        "description": (
            "Model linear yang digunakan untuk memprediksi "
            "probabilitas kelas pada masalah classification."
        ),
        "strengths": [
            "Baseline yang kuat untuk classification.",
            "Relatif mudah diinterpretasikan.",
            "Cepat dilatih.",
            "Cocok untuk hubungan yang relatif linear.",
        ],
        "limitations": [
            "Kurang cocok untuk pola yang sangat nonlinear.",
            "Sensitif terhadap feature scaling.",
            "Dapat terpengaruh oleh multikolinearitas.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu dilakukan encoding.",
            "Feature scaling biasanya disarankan.",
        ],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "decision_tree_classifier": {
        "name": "Decision Tree Classifier",
        "category": "Classification",
        "suitable_tasks": [
            "binary_classification",
            "multiclass_classification",
        ],
        "description": (
            "Model berbasis aturan keputusan yang membagi "
            "data menjadi beberapa kelompok berdasarkan fitur."
        ),
        "strengths": [
            "Mudah diinterpretasikan.",
            "Tidak membutuhkan feature scaling.",
            "Dapat menangani hubungan nonlinear.",
            "Dapat divisualisasikan dalam bentuk tree.",
        ],
        "limitations": [
            "Mudah mengalami overfitting.",
            "Perubahan kecil pada data dapat mengubah struktur tree.",
            "Tree yang terlalu dalam dapat menjadi kompleks.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak diperlukan.",
        ],
        "interpretability": "High",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "category": "Classification",
        "suitable_tasks": [
            "binary_classification",
            "multiclass_classification",
        ],
        "description": (
            "Ensemble dari banyak decision tree untuk "
            "meningkatkan stabilitas dan kemampuan prediksi."
        ),
        "strengths": [
            "Mampu menangani hubungan nonlinear.",
            "Relatif robust terhadap variasi data.",
            "Dapat digunakan untuk feature importance.",
            "Biasanya tidak membutuhkan feature scaling.",
        ],
        "limitations": [
            "Lebih kompleks dibandingkan model linear.",
            "Model dapat menggunakan lebih banyak resource.",
            "Interpretasi individual prediction tidak sesederhana decision tree.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak diperlukan.",
        ],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "svm_classifier": {
        "name": "Support Vector Machine (SVM)",
        "category": "Classification",
        "suitable_tasks": [
            "binary_classification",
            "multiclass_classification",
        ],
        "description": (
            "Metode yang mencari decision boundary optimal "
            "untuk memisahkan kelas."
        ),
        "strengths": [
            "Efektif pada dataset kecil hingga menengah.",
            "Dapat menangani boundary nonlinear menggunakan kernel.",
            "Cocok untuk feature space berdimensi tinggi.",
        ],
        "limitations": [
            "Training dapat menjadi mahal pada dataset sangat besar.",
            "Sensitif terhadap feature scaling.",
            "Pemilihan kernel dan parameter perlu diperhatikan.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling sangat disarankan.",
        ],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    "knn_classifier": {
        "name": "K-Nearest Neighbors (KNN)",
        "category": "Classification",
        "suitable_tasks": [
            "binary_classification",
            "multiclass_classification",
        ],
        "description": (
            "Metode yang menentukan kelas berdasarkan "
            "tetangga terdekat dari suatu observasi."
        ),
        "strengths": [
            "Konsep sederhana.",
            "Tidak membutuhkan proses training model yang kompleks.",
            "Dapat menangkap pola lokal pada data.",
        ],
        "limitations": [
            "Prediction dapat menjadi lambat pada dataset besar.",
            "Sensitif terhadap feature scaling.",
            "Dapat terpengaruh oleh dimensionalitas tinggi.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling sangat disarankan.",
        ],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    # ============================================================
    # REGRESSION
    # ============================================================

    "linear_regression": {
        "name": "Linear Regression",
        "category": "Regression",
        "suitable_tasks": [
            "regression",
        ],
        "description": (
            "Model linear untuk memprediksi nilai target kontinu."
        ),
        "strengths": [
            "Sederhana dan mudah diinterpretasikan.",
            "Cocok sebagai baseline regression.",
            "Cepat dilatih.",
        ],
        "limitations": [
            "Mengasumsikan hubungan linear.",
            "Sensitif terhadap outlier.",
            "Dapat terpengaruh oleh multikolinearitas.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling dapat membantu pada kondisi tertentu.",
        ],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "decision_tree_regressor": {
        "name": "Decision Tree Regressor",
        "category": "Regression",
        "suitable_tasks": [
            "regression",
        ],
        "description": (
            "Decision tree yang digunakan untuk memprediksi "
            "nilai numerik kontinu."
        ),
        "strengths": [
            "Mampu menangani hubungan nonlinear.",
            "Tidak membutuhkan feature scaling.",
            "Mudah divisualisasikan.",
            "Relatif mudah diinterpretasikan.",
        ],
        "limitations": [
            "Mudah mengalami overfitting.",
            "Sensitif terhadap perubahan data.",
            "Tree yang terlalu dalam dapat menjadi kompleks.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak diperlukan.",
        ],
        "interpretability": "High",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "category": "Regression",
        "suitable_tasks": [
            "regression",
        ],
        "description": (
            "Ensemble decision tree yang digunakan "
            "untuk memprediksi nilai numerik kontinu."
        ),
        "strengths": [
            "Mampu menangani hubungan nonlinear.",
            "Relatif robust terhadap variasi data.",
            "Dapat digunakan untuk feature importance.",
            "Tidak membutuhkan feature scaling.",
        ],
        "limitations": [
            "Lebih kompleks dibandingkan linear regression.",
            "Menggunakan lebih banyak resource.",
            "Interpretasi model tidak sesederhana model linear.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak diperlukan.",
        ],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "category": "Regression",
        "suitable_tasks": [
            "regression",
        ],
        "description": (
            "Ensemble model yang membangun model secara "
            "bertahap untuk memperbaiki error sebelumnya."
        ),
        "strengths": [
            "Kuat untuk pola nonlinear.",
            "Sering efektif pada data tabular.",
            "Dapat menghasilkan performa tinggi setelah tuning.",
        ],
        "limitations": [
            "Training dapat lebih lama.",
            "Sensitif terhadap hyperparameter.",
            "Dapat mengalami overfitting jika tidak dikontrol.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak diperlukan.",
        ],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    # ============================================================
    # CLUSTERING
    # ============================================================

    "kmeans": {
        "name": "K-Means",
        "category": "Clustering",
        "suitable_tasks": [
            "clustering",
        ],
        "description": (
            "Algoritma clustering yang mengelompokkan data "
            "berdasarkan kedekatan terhadap centroid."
        ),
        "strengths": [
            "Sederhana dan populer.",
            "Relatif cepat.",
            "Mudah divisualisasikan.",
        ],
        "limitations": [
            "Jumlah cluster perlu ditentukan.",
            "Sensitif terhadap outlier.",
            "Hasil dipengaruhi oleh skala fitur.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Feature scaling sangat disarankan.",
            "Categorical features perlu preprocessing khusus.",
        ],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "dbscan": {
        "name": "DBSCAN",
        "category": "Clustering",
        "suitable_tasks": [
            "clustering",
        ],
        "description": (
            "Clustering berbasis kepadatan yang dapat "
            "menemukan cluster dengan bentuk tidak beraturan."
        ),
        "strengths": [
            "Tidak selalu membutuhkan jumlah cluster di awal.",
            "Dapat mendeteksi noise.",
            "Dapat menemukan cluster dengan bentuk tidak beraturan.",
        ],
        "limitations": [
            "Sensitif terhadap parameter epsilon dan minimum samples.",
            "Kurang cocok jika kepadatan cluster sangat berbeda.",
            "Dapat terpengaruh oleh dimensionalitas tinggi.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Feature scaling sangat disarankan.",
            "Categorical features perlu preprocessing.",
        ],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "agglomerative_clustering": {
        "name": "Agglomerative Clustering",
        "category": "Clustering",
        "suitable_tasks": [
            "clustering",
        ],
        "description": (
            "Hierarchical clustering yang membangun "
            "kelompok data secara bertahap."
        ),
        "strengths": [
            "Dapat menghasilkan struktur hierarki cluster.",
            "Dapat divisualisasikan menggunakan dendrogram.",
            "Berguna untuk eksplorasi struktur data.",
        ],
        "limitations": [
            "Dapat mahal pada dataset besar.",
            "Pemilihan linkage memengaruhi hasil.",
            "Sensitif terhadap distance metric.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Feature scaling disarankan.",
            "Categorical features perlu preprocessing.",
        ],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    # ============================================================
    # ANOMALY DETECTION
    # ============================================================

    "isolation_forest": {
        "name": "Isolation Forest",
        "category": "Anomaly Detection",
        "suitable_tasks": [
            "anomaly_detection",
        ],
        "description": (
            "Algoritma yang mendeteksi observasi yang "
            "lebih mudah diisolasi dari observasi lainnya."
        ),
        "strengths": [
            "Efektif untuk anomaly detection pada data tabular.",
            "Dapat menangani dataset dengan banyak fitur.",
            "Tidak membutuhkan label anomaly.",
            "Dapat digunakan pada dataset relatif besar.",
        ],
        "limitations": [
            "Hasil dipengaruhi oleh contamination dan parameter model.",
            "Anomaly score membutuhkan interpretasi yang tepat.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling umumnya tidak wajib.",
        ],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "local_outlier_factor": {
        "name": "Local Outlier Factor (LOF)",
        "category": "Anomaly Detection",
        "suitable_tasks": [
            "anomaly_detection",
        ],
        "description": (
            "Metode yang mendeteksi anomaly berdasarkan "
            "kepadatan lokal suatu observasi."
        ),
        "strengths": [
            "Memperhatikan struktur lokal data.",
            "Berguna ketika kepadatan antar wilayah berbeda.",
            "Tidak membutuhkan label anomaly.",
        ],
        "limitations": [
            "Kurang cocok untuk dataset sangat besar.",
            "Sensitif terhadap jumlah neighbors.",
            "Dapat terpengaruh oleh dimensionalitas tinggi.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Feature scaling sangat disarankan.",
            "Categorical features perlu preprocessing.",
        ],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    "one_class_svm": {
        "name": "One-Class SVM",
        "category": "Anomaly Detection",
        "suitable_tasks": [
            "anomaly_detection",
        ],
        "description": (
            "Metode berbasis SVM untuk mempelajari wilayah "
            "normal dan mendeteksi observasi di luar wilayah tersebut."
        ),
        "strengths": [
            "Dapat digunakan tanpa label anomaly.",
            "Dapat menangani boundary nonlinear menggunakan kernel.",
        ],
        "limitations": [
            "Training dapat mahal pada dataset besar.",
            "Sensitif terhadap parameter.",
            "Feature scaling penting.",
        ],
        "preprocessing": [
            "Missing values perlu ditangani.",
            "Categorical features perlu encoding.",
            "Feature scaling sangat disarankan.",
        ],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    "elliptic_envelope": {
        "name": "Elliptic Envelope",
        "category": "Anomaly Detection",
        "suitable_tasks": ["anomaly_detection"],
        "description": "Metode estimasi kovariansi Gaussian untuk mendeteksi outlier pada data berdistribusi normal.",
        "strengths": ["Cepat dan efektif untuk data terdistribusi normal.", "Menghasilkan batas elips yang jelas."],
        "limitations": ["Mengasumsikan distribusi Gaussian.", "Sensitif terhadap data multimodal."],
        "preprocessing": ["Feature scaling disarankan.", "Missing values perlu diimputasi."],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "extra_trees_classifier": {
        "name": "Extra Trees Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Ensemble decision tree dengan pemisahan threshold yang sangat acak untuk mengurangi variansi.",
        "strengths": ["Sering lebih cepat dari Random Forest.", "Mengurangi risiko overfitting."],
        "limitations": ["Menggunakan resource komputasi signifikan pada dataset sangat besar."],
        "preprocessing": ["Missing values perlu ditangani.", "Feature scaling tidak wajib."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "hist_gradient_boosting_classifier": {
        "name": "HistGradientBoosting Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Gradient boosting berbasis histogram terinspirasi oleh LightGBM, sangat cepat untuk dataset besar.",
        "strengths": ["Sangat cepat dan efisien memori.", "Mendukung missing values bawaan."],
        "limitations": ["Kurang cocok untuk dataset berukuran sangat kecil."],
        "preprocessing": ["Missing values dapat ditangani otomatis.", "Encoding kategorikal disarankan."],
        "interpretability": "Low",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "adaboost_classifier": {
        "name": "AdaBoost Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Boosting sekuensial yang memberikan bobot lebih pada observasi yang salah diprediksi.",
        "strengths": ["Efektif pada dataset berdimensi kecil hingga menengah.", "Mudah dikonfigurasi."],
        "limitations": ["Sensitif terhadap noise dan outlier."],
        "preprocessing": ["Outlier perlu ditangani.", "Missing values wajib diimputasi."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "gaussian_nb": {
        "name": "Gaussian Naive Bayes",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Model probabilistik berbasis teorema Bayes dengan asumsi independensi fitur Gaussian.",
        "strengths": ["Sangat cepat dilatih dan diprediksi.", "Baseline yang andal untuk data berdimensi tinggi."],
        "limitations": ["Mengasumsikan independensi fitur yang kuat."],
        "preprocessing": ["Missing values perlu ditangani.", "Distribusi normal membantu."],
        "interpretability": "High",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "linear_discriminant_analysis": {
        "name": "Linear Discriminant Analysis (LDA)",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Mencari kombinasi linear fitur yang memaksimalkan pemisahan antar kelas.",
        "strengths": ["Sederhana, analitis, dan bebas tuning parameter."],
        "limitations": ["Mengasumsikan kovariansi kelas yang sama."],
        "preprocessing": ["Feature scaling disarankan.", "Missing values wajib diimputasi."],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "mlp_classifier": {
        "name": "Multi-Layer Perceptron Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Jaringan saraf tiruan feedforward dengan representasi non-linear berlapis.",
        "strengths": ["Mampu memodelkan hubungan non-linear yang sangat kompleks."],
        "limitations": ["Membutuhkan data berskala.", "Waktu training lebih lama."],
        "preprocessing": ["Feature scaling wajib.", "Missing values wajib diimputasi."],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "xgboost_classifier": {
        "name": "XGBoost Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Pustaka Extreme Gradient Boosting yang sangat populer untuk tabular data.",
        "strengths": ["Performa tinggi pada kompetisi data science.", "Regularisasi bawaan."],
        "limitations": ["Memerlukan instalasi library eksternal 'xgboost'."],
        "preprocessing": ["Missing values ditangani otomatis.", "Encoding kategorikal disarankan."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "lightgbm_classifier": {
        "name": "LightGBM Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Framework gradient boosting berbasis tree dengan leaf-wise growth dan kecepatan tinggi.",
        "strengths": ["Kecepatan training luar biasa.", "Mendukung categorical feature bawaan."],
        "limitations": ["Memerlukan instalasi library eksternal 'lightgbm'."],
        "preprocessing": ["Missing values ditangani otomatis."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "catboost_classifier": {
        "name": "CatBoost Classifier",
        "category": "Classification",
        "suitable_tasks": ["binary_classification", "multiclass_classification"],
        "description": "Gradient boosting dengan penanganan fitur kategorikal canggih tanpa overfitting.",
        "strengths": ["Optimal untuk dataset dengan banyak fitur kategorikal."],
        "limitations": ["Memerlukan instalasi library eksternal 'catboost'."],
        "preprocessing": ["Mendukung kategorikal langsung."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "ridge_regression": {
        "name": "Ridge Regression",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Regresi linear dengan regularisasi L2 untuk mengatasi multikolinearitas.",
        "strengths": ["Mengurangi variansi bobot model.", "Sangat stabil pada data berkolerasi."],
        "limitations": ["Tidak melakukan seleksi fitur."],
        "preprocessing": ["Feature scaling wajib.", "Missing values wajib diimputasi."],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "lasso_regression": {
        "name": "Lasso Regression",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Regresi linear dengan penalti L1 yang mampu mengnolkan koefisien fitur yang tidak relevan.",
        "strengths": ["Melakukan seleksi fitur otomatis.", "Model sederhana dan interpretable."],
        "limitations": ["Dapat memilih secara arbitrer jika fitur sangat berkorelasi."],
        "preprocessing": ["Feature scaling wajib.", "Missing values wajib diimputasi."],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "elasticnet_regression": {
        "name": "ElasticNet Regression",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Regresi linear yang menggabungkan penalti L1 dan L2 secara simultan.",
        "strengths": ["Menggabungkan keunggulan seleksi fitur Lasso dan stabilitas Ridge."],
        "limitations": ["Memerlukan tuning parameter rasio L1."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "extra_trees_regressor": {
        "name": "Extra Trees Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Ensemble pohon regresi dengan pemotongan ambang acak ekstrim.",
        "strengths": ["Mengurangi variansi dan lebih cepat daripada Random Forest."],
        "limitations": ["Penggunaan memori lebih tinggi."],
        "preprocessing": ["Feature scaling umumnya tidak diperlukan."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "hist_gradient_boosting_regressor": {
        "name": "HistGradientBoosting Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Gradient boosting regressor berbasis bin histogram untuk kecepatan maksimum.",
        "strengths": ["Sangat cepat pada data besar.", "Menangani missing values."],
        "limitations": ["Kurang optimal pada dataset kecil (< 100 baris)."],
        "preprocessing": ["Missing values ditangani otomatis."],
        "interpretability": "Low",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "adaboost_regressor": {
        "name": "AdaBoost Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Boosting regresi yang menyesuaikan bobot sampel berdasarkan error prediksi.",
        "strengths": ["Meningkatkan akurasi regresi dasar."],
        "limitations": ["Sensitif terhadap outlier nilai kontinu."],
        "preprocessing": ["Outlier perlu ditangani."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "huber_regressor": {
        "name": "Huber Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Regresi linear yang menggunakan fungsi kerugian Huber sehingga kebal terhadap outlier.",
        "strengths": ["Sangat robust terhadap keberadaan outlier numerik."],
        "limitations": ["Hanya memodelkan hubungan linear."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "High",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": True,
        "large_data": True,
    },

    "svr": {
        "name": "Support Vector Regressor (SVR)",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Support vector machine untuk regresi dengan margin toleransi epsilon.",
        "strengths": ["Efektif pada dataset dengan non-linearitas kompleks."],
        "limitations": ["Komputasi mahal pada dataset besar ($O(N^3)$)."],
        "preprocessing": ["Feature scaling sangat penting."],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },

    "mlp_regressor": {
        "name": "Multi-Layer Perceptron Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Jaringan saraf tiruan feedforward untuk memprediksi nilai kontinu.",
        "strengths": ["Mampu memodelkan fungsi non-linear kontinu yang rumit."],
        "limitations": ["Sensitif terhadap skala fitur dan inisialisasi bobot."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Extreme Gradient Boosting regressor berkinerja tinggi.",
        "strengths": ["Sering menjadi pemenang kompetisi data tabular."],
        "limitations": ["Memerlukan library eksternal 'xgboost'."],
        "preprocessing": ["Missing values ditangani bawaan."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "lightgbm_regressor": {
        "name": "LightGBM Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Gradient boosting regressor cepat berbasis histogram.",
        "strengths": ["Kecepatan komputasi tinggi pada data besar."],
        "limitations": ["Memerlukan library eksternal 'lightgbm'."],
        "preprocessing": ["Missing values ditangani bawaan."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": False,
        "large_data": True,
    },

    "catboost_regressor": {
        "name": "CatBoost Regressor",
        "category": "Regression",
        "suitable_tasks": ["regression"],
        "description": "Gradient boosting regressor dengan penanganan kategorikal optimal.",
        "strengths": ["Performa tinggi pada data campuran numerik-kategorikal."],
        "limitations": ["Memerlukan library eksternal 'catboost'."],
        "preprocessing": ["Mendukung kategorikal langsung."],
        "interpretability": "Medium",
        "scaling_required": False,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "minibatch_kmeans": {
        "name": "Mini-Batch K-Means",
        "category": "Clustering",
        "suitable_tasks": ["clustering"],
        "description": "Varian K-Means yang menggunakan mini-batch acak untuk mempercepat clustering pada dataset besar.",
        "strengths": ["Jauh lebih cepat daripada K-Means standar pada data besar.", "Penggunaan memori rendah."],
        "limitations": ["Kualitas klaster sedikit lebih rendah dari standar K-Means."],
        "preprocessing": ["Feature scaling wajib.", "Missing values wajib diimputasi."],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": False,
        "large_data": True,
    },

    "gaussian_mixture": {
        "name": "Gaussian Mixture Model",
        "category": "Clustering",
        "suitable_tasks": ["clustering"],
        "description": "Model clustering probabilistik berbasis campuran distribusi Gaussian multivariat.",
        "strengths": ["Menghasilkan keanggotaan klaster lunak (soft clustering).", "Mampu menangani bentuk elips."],
        "limitations": ["Sensitif terhadap inisialisasi lokal."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": True,
    },

    "birch": {
        "name": "BIRCH",
        "category": "Clustering",
        "suitable_tasks": ["clustering"],
        "description": "Clustering hierarkis bertingkat menggunakan Clustering Feature Tree yang sangat hemat memori.",
        "strengths": ["Hanya memerlukan 1 lintasan pada dataset besar."],
        "limitations": ["Hanya cocok untuk fitur numerik berbasis metrik jarak."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "Medium",
        "scaling_required": True,
        "nonlinear": False,
        "small_data": False,
        "large_data": True,
    },

    "spectral_clustering": {
        "name": "Spectral Clustering",
        "category": "Clustering",
        "suitable_tasks": ["clustering"],
        "description": "Clustering berbasis graf dan nilai eigen matriks similaritas untuk struktur manifold rumit.",
        "strengths": ["Mampu memisahkan klaster dengan topologi non-linear kompleks."],
        "limitations": ["Sangat mahal secara komputasi ($O(N^3)$)."],
        "preprocessing": ["Feature scaling wajib."],
        "interpretability": "Low",
        "scaling_required": True,
        "nonlinear": True,
        "small_data": True,
        "large_data": False,
    },
}


def get_method_info(method_id: str) -> dict | None:
    """
    Mengambil informasi sebuah method berdasarkan method_id.
    """

    return METHOD_INFO.get(method_id)


def get_all_methods() -> dict:
    """
    Mengembalikan seluruh knowledge base method.
    """

    return METHOD_INFO.copy()


def get_methods_for_task(task: str) -> dict:
    """
    Mengambil semua method yang cocok dengan sebuah ML task.
    """

    return {
        method_id: info
        for method_id, info in METHOD_INFO.items()
        if task in info.get("suitable_tasks", [])
    }