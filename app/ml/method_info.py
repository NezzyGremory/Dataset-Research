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