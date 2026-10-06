from __future__ import annotations

from typing import Any, Dict, List, Optional


class LocalAcademicExplainer:
    """
    Generator narasi analitis akademis lokal (100% offline & deterministik).

    Menyusun sintesis interpretasi dataset ilmiah dalam Bahasa Indonesia formal
    berdasarkan fakta empiris yang telah dihitung oleh profiler lokal.
    Tidak memerlukan API key dan tidak memiliki risiko rate-limit.
    """

    def explain(self, analysis: Dict[str, Any]) -> str:
        if not isinstance(analysis, dict):
            return "Informasi analisis dataset tidak valid untuk menyusun narasi interpretasi."

        profile = analysis.get("profile") or {}
        statistics = analysis.get("statistics") or []
        missing_values = analysis.get("missing_values") or []
        duplicates = analysis.get("duplicates") or {}
        outliers = analysis.get("outliers") or []
        correlations = analysis.get("correlations")
        fingerprint = analysis.get("fingerprint") or {}

        paragraphs: List[str] = []

        # 1. Gambaran Umum & Karakteristik Dimensi
        p1 = self._build_overview_paragraph(profile, fingerprint)
        if p1:
            paragraphs.append(p1)

        # 2. Kandidat Target & Tugas Machine Learning
        p2 = self._build_target_paragraph(fingerprint, statistics)
        if p2:
            paragraphs.append(p2)

        # 3. Kualitas Data (Nilai Kosong & Duplikasi Observasi)
        p3 = self._build_quality_paragraph(profile, missing_values, duplicates)
        if p3:
            paragraphs.append(p3)

        # 4. Deteksi Outlier & Karakteristik Distribusi
        p4 = self._build_outliers_paragraph(outliers)
        if p4:
            paragraphs.append(p4)

        # 5. Analisis Korelasi & Hubungan Bivariat
        p5 = self._build_correlation_paragraph(correlations)
        if p5:
            paragraphs.append(p5)

        # 6. Kesimpulan Metodologis & Kesiapan Riset
        p6 = self._build_conclusion_paragraph(profile, missing_values, duplicates, outliers)
        if p6:
            paragraphs.append(p6)

        return "\n\n".join(paragraphs)

    # =========================================================================
    # PARAGRAPH BUILDERS
    # =========================================================================

    def _build_overview_paragraph(self, profile: Dict[str, Any], fingerprint: Dict[str, Any]) -> str:
        rows = profile.get("rows", profile.get("row_count", 0))
        cols = profile.get("columns", profile.get("column_count", 0))
        num_cols = profile.get("numeric_columns", 0)
        cat_cols = profile.get("categorical_columns", 0)
        mem_bytes = profile.get("memory_usage", 0)

        mem_str = (
            f"{mem_bytes / 1024:.1f} KB"
            if mem_bytes < 1024 * 1024
            else f"{mem_bytes / (1024 * 1024):.2f} MB"
        )

        fp_hash = fingerprint.get("fingerprint")
        hash_note = f" dengan fingerprint integritas SHA-256 ({fp_hash[:16]}...)" if fp_hash else ""

        if rows == 0 and cols == 0:
            return "Dataset belum memuat data observasi yang cukup untuk dianalisis."

        scale_desc = (
            "berskala kecil hingga menengah (small-to-medium tabular dataset)"
            if rows < 10000
            else "berskala besar dengan volume sampel yang representatif"
        )

        return (
            f"Berdasarkan profiling struktural, dataset ini memuat sebanyak {rows:,} baris observasi "
            f"dan {cols:,} atribut fitur{hash_note}. Struktur dataset tergolong {scale_desc}, "
            f"dengan komposisi {num_cols:,} fitur bertipe numerik dan {cat_cols:,} fitur bertipe kategorikal/objek. "
            f"Alokasi memori kerja tercatat sekitar {mem_str}, menunjukkan bahwa pemrosesan in-memory dan "
            f"pelatihan algoritma machine learning standar dapat dieksekusi secara efisien tanpa kendala komputasi."
        )

    def _build_target_paragraph(self, fingerprint: Dict[str, Any], statistics: List[Dict[str, Any]]) -> str:
        rep = fingerprint.get("representation", {}) if isinstance(fingerprint, dict) else {}
        target_candidates = rep.get("target_candidates", [])
        col_sigs = {c.get("name"): c for c in rep.get("column_signature", []) if isinstance(c, dict)}

        if target_candidates:
            target = target_candidates[0]
            sig = col_sigs.get(target, {})
            u_count = sig.get("unique_count")

            if u_count == 2:
                task_name = "Klasifikasi Biner (Binary Classification)"
                task_desc = "memprediksi probabilitas keterjadian dari dua kelas luaran yang saling eksklusif"
            elif isinstance(u_count, int) and u_count <= 10:
                task_name = f"Klasifikasi Multikelas ({u_count} Kelas)"
                task_desc = "mengelompokkan observasi ke dalam kategori diskret jamak"
            else:
                task_name = "Regresi / Estimasi Kontinu"
                task_desc = "memodelkan hubungan linear maupun non-linear terhadap nilai target numerik kontinu"

            score = sig.get("target_score", 60)
            return (
                f"Dalam konteks formulasi masalah penelitian, sistem mendeteksi fitur '{target}' sebagai "
                f"kandidat target utama dengan skor kelayakan heuristik sebesar {score}/100. Karakteristik "
                f"tersebut mengindikasikan bahwa dataset sangat ideal untuk task {task_name}, yakni {task_desc}. "
                f"Fitur-fitur lain yang tersisa dapat diposisikan sebagai variabel independen (prediktor) setelah "
                f"melewati tahapan seleksi dan transformasi fitur yang relevan."
            )
        else:
            return (
                "Berdasarkan inspeksi skema data, tidak ditemukan kolom target berlabel eksplisit yang dominan. "
                "Hal ini membuka peluang penelitian berbasis Unsupervised Learning, seperti analisis klaster "
                "(Clustering menggunakan K-Means atau Hierarchical Clustering) untuk segmentasi pola data laten, "
                "reduksi dimensi melalui Principal Component Analysis (PCA), atau pemodelan deteksi anomali."
            )

    def _build_quality_paragraph(
        self,
        profile: Dict[str, Any],
        missing_values: List[Dict[str, Any]],
        duplicates: Dict[str, Any],
    ) -> str:
        rows = profile.get("rows", 0)
        dup_count = int(duplicates.get("duplicate_count", 0)) if isinstance(duplicates, dict) else 0

        active_missing = [
            m for m in missing_values
            if isinstance(m, dict) and float(m.get("missing_count", 0)) > 0
        ]

        quality_notes = []

        # Missing values notes
        if active_missing:
            col_details = []
            for item in sorted(active_missing, key=lambda x: float(x.get("missing_count", 0)), reverse=True)[:3]:
                col_name = item.get("column", "kolom")
                pct = float(item.get("missing_percentage", 0))
                col_details.append(f"'{col_name}' ({pct:.1f}%)")

            details_str = ", ".join(col_details)
            quality_notes.append(
                f"Terdapat nilai kosong (missing values) yang terdeteksi pada {len(active_missing)} atribut, "
                f"dengan konsentrasi tertinggi pada fitur {details_str}. Fitur dengan proporsi kehilangan data ekstrem "
                f"memerlukan evaluasi seleksi fitur (drop feature) atau imputasi terkontrol berbasis median/modus "
                f"guna memitigasi risiko bias estimasi pada pemodelan."
            )
        else:
            quality_notes.append(
                "Keutuhan data tergolong sangat prima dengan kelengkapan 100% tanpa adanya nilai kosong (missing values) "
                "pada seluruh kolom observasi."
            )

        # Duplicates notes
        if dup_count > 0:
            dup_pct = (dup_count / rows * 100) if rows > 0 else 0
            quality_notes.append(
                f"Selain itu, teridentifikasi sebanyak {dup_count:,} baris duplikat identik ({dup_pct:.2f}%). "
                f"Pembersihan duplikasi sebelum proses pembagian data (train-test split) sangat direkomendasikan "
                f"untuk mencegah risiko data leakage dan overestimasi performa evaluasi model."
            )
        else:
            quality_notes.append(
                "Tidak ditemukan baris duplikat identik di dalam dataset, sehingga seluruh sampel merupakan observasi unik."
            )

        return " ".join(quality_notes)

    def _build_outliers_paragraph(self, outliers: List[Dict[str, Any]]) -> str:
        active_outliers = [
            o for o in outliers
            if isinstance(o, dict) and float(o.get("outlier_count", 0)) > 0
        ]

        if not active_outliers:
            return (
                "Uji sebaran nilai ekstrem menggunakan metode Interquartile Range (IQR) dengan ambang batas 1.5×IQR "
                "menunjukkan bahwa seluruh fitur numerik terdistribusi secara wajar tanpa terdeteksi pencilan ekstrem. "
                "Kondisi ini meminimalkan distorsi pada fungsi objektif model berbasis jarak maupun regresi linear."
            )

        top_outliers = sorted(active_outliers, key=lambda x: float(x.get("outlier_count", 0)), reverse=True)[:3]
        outlier_strs = [
            f"'{o.get('column')}' ({int(o.get('outlier_count', 0)):,} observasi, {float(o.get('outlier_percentage', 0)):.1f}%)"
            for o in top_outliers
        ]

        return (
            f"Analisis dispersi menggunakan metode Interquartile Range (IQR) mengidentifikasi adanya titik data "
            f"di luar batas normal pada {len(active_outliers)} fitur numerik, khususnya pada {', '.join(outlier_strs)}. "
            f"Dalam metodologi penelitian ilmiah, outlier ini tidak serta-merta dianggap sebagai galat (noise), "
            f"melainkan representasi variabilitas alami fenomena empiris. Penggunaan estimator yang kokoh "
            f"(RobustScaler) atau algoritma berbasis pohon keputusan (seperti Random Forest dan Gradient Boosting) "
            f"sangat dianjurkan karena memiliki invariansi alami terhadap keberadaan pencilan."
        )

    def _build_correlation_paragraph(self, correlations: Any) -> str:
        pairs = []
        if hasattr(correlations, "columns") and hasattr(correlations, "iloc"):
            try:
                cols = list(correlations.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        val = float(correlations.iloc[i, j])
                        if val == val:
                            pairs.append((abs(val), cols[i], cols[j], val))
                pairs.sort(reverse=True)
            except Exception:
                pass

        if not pairs:
            return (
                "Dataset tidak memuat jumlah pasangan fitur numerik yang memadai untuk melakukan komputasi "
                "korelasi bivariat Pearson secara komprehensif."
            )

        top_pair = pairs[0]
        val = top_pair[3]
        c1, c2 = top_pair[1], top_pair[2]
        direction = "positif" if val > 0 else "negatif"
        strength = "kuat" if abs(val) >= 0.7 else ("moderat/sedang" if abs(val) >= 0.3 else "lemah")

        multicollinear = any(p[0] > 0.85 for p in pairs)

        corr_text = (
            f"Investigasi asosiasi linear antar-fitur memperlihatkan korelasi terkuat antara '{c1}' dan '{c2}' "
            f"dengan koefisien Pearson r = {val:+.2f} (hubungan {direction} berkekuatan {strength}). "
        )

        if multicollinear:
            corr_text += (
                "Ditemukan pasangan fitur dengan korelasi tinggi (|r| > 0.85) yang berpotensi memicu isu "
                "multikolinearitas. Pada tahap feature engineering, disarankan menerapkan teknik seleksi fitur "
                "atau regularisasi (Lasso/Ridge) untuk menjaga kestabilan koefisien estimasi."
            )
        else:
            corr_text += (
                "Secara keseluruhan, tidak terdeteksi multikolinearitas ekstrem (|r| > 0.85), mengindikasikan "
                "bahwa masing-masing prediktor menyumbang informasi variansi yang unik dan independen."
            )

        return corr_text

    def _build_conclusion_paragraph(
        self,
        profile: Dict[str, Any],
        missing_values: List[Dict[str, Any]],
        duplicates: Dict[str, Any],
        outliers: List[Dict[str, Any]],
    ) -> str:
        has_missing = any(float(m.get("missing_count", 0)) > 0 for m in missing_values if isinstance(m, dict))
        has_dup = int(duplicates.get("duplicate_count", 0) if isinstance(duplicates, dict) else 0) > 0

        steps = []
        if has_dup:
            steps.append("eliminasi baris duplikat")
        if has_missing:
            steps.append("penanganan missing values melalui strategi imputasi yang terjustifikasi")
        steps.append("standarisasi skala fitur numerik")
        steps.append("evaluasi model benchmark machine learning")

        prep_roadmap = ", ".join(steps)

        return (
            f"Sebagai sintesis kesimpulan, dataset ini berada pada kondisi yang siap untuk ditindaklanjuti ke alur "
            f"penelitian eksperimental. Roadmap metodologis berikutnya mencakup {prep_roadmap}. "
            f"Dengan menerapkan prinsip reproducibility melalui Dataset Versions & Research Trail, "
            f"setiap tahapan transformasi data akan tercatat secara transparan dan dapat dipertanggungjawabkan "
            f"secara akademis."
        )
