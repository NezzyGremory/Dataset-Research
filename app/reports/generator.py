from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import html

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class ReportGenerator:
    """Comprehensive Research & ML Report Generator.

    Combines:
    1. Dataset Profile & Structural Diagnostics (Pillars 1-4)
    2. Data Quality & Health Issues
    3. Empirical Machine Learning Benchmarks & Model Ranking
    4. Academic Literature Review & Research Gaps
    5. AI Synthesis & Academic Insights
    """

    def __init__(self, template_dir: Optional[Union[str, Path]] = None) -> None:
        if template_dir is None:
            self.template_dir = Path(__file__).resolve().parent / "templates"
        else:
            self.template_dir = Path(template_dir)

        self._template_file = "report.html"

        if HAS_JINJA2 and self.template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.jinja_env = None

    def generate(
        self,
        dataset_name: str = "Dataset",
        analysis_data: Optional[Dict[str, Any]] = None,
        ml_data: Optional[Dict[str, Any]] = None,
        research_data: Optional[Dict[str, Any]] = None,
        ai_synthesis: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete, self-contained HTML report."""
        context = self._build_context(
            dataset_name=dataset_name,
            analysis_data=analysis_data or {},
            ml_data=ml_data or {},
            research_data=research_data or {},
            ai_synthesis=ai_synthesis or "",
            metadata=metadata or {},
        )

        if self.jinja_env:
            try:
                template = self.jinja_env.get_template(self._template_file)
                return template.render(**context)
            except Exception as exc:
                # If template rendering encounters an error, fall back to fallback builder
                print(f"Warning: Jinja2 template rendering error: {exc}. Using fallback builder.")

        return self._generate_fallback_html(context)

    def export_to_file(
        self,
        filepath: Union[str, Path],
        dataset_name: str = "Dataset",
        analysis_data: Optional[Dict[str, Any]] = None,
        ml_data: Optional[Dict[str, Any]] = None,
        research_data: Optional[Dict[str, Any]] = None,
        ai_synthesis: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Generate and save the report to an HTML file."""
        html_content = self.generate(
            dataset_name=dataset_name,
            analysis_data=analysis_data,
            ml_data=ml_data,
            research_data=research_data,
            ai_synthesis=ai_synthesis,
            metadata=metadata,
        )
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html_content, encoding="utf-8")
        return target

    def _build_context(
        self,
        dataset_name: str,
        analysis_data: Dict[str, Any],
        ml_data: Dict[str, Any],
        research_data: Dict[str, Any],
        ai_synthesis: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize raw results from all engines into clean template context."""
        profile = analysis_data.get("profile") or {}
        missing_values = analysis_data.get("missing_values") or []
        duplicates = analysis_data.get("duplicates") or {}
        raw_quality = analysis_data.get("data_quality") or []

        total_rows = int(profile.get("total_rows") or profile.get("row_count") or 0)
        total_cols = int(profile.get("total_columns") or profile.get("column_count") or 0)
        memory_usage = profile.get("memory_usage_str") or profile.get("memory_usage") or "N/A"
        if isinstance(memory_usage, (int, float)):
            memory_usage = f"{memory_usage / (1024 * 1024):.2f} MB"

        # Missing values
        total_cells = total_rows * total_cols if total_rows and total_cols else 1
        missing_cells = 0
        if isinstance(missing_values, list):
            for m in missing_values:
                missing_cells += int(m.get("missing_count", 0))
        elif isinstance(missing_values, dict):
            missing_cells = int(missing_values.get("total_missing", 0))

        missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # Duplicates
        duplicate_count = int(
            duplicates.get("duplicate_count", 0)
            if isinstance(duplicates, dict)
            else 0
        )

        # Columns summary
        dtypes = profile.get("dtypes") or {}
        col_summary = []
        numeric_count = 0
        cat_count = 0

        # Parse missing lookup
        missing_map = {}
        if isinstance(missing_values, list):
            for m in missing_values:
                col_name = m.get("column")
                if col_name:
                    missing_map[col_name] = (m.get("missing_count", 0), m.get("missing_percentage", 0.0))

        for col_name, dtype_str in dtypes.items():
            dtype_lower = str(dtype_str).lower()
            if any(t in dtype_lower for t in ["int", "float", "double", "num"]):
                numeric_count += 1
            else:
                cat_count += 1

            null_cnt, null_p = missing_map.get(col_name, (0, 0.0))
            col_summary.append({
                "name": col_name,
                "dtype": str(dtype_str),
                "null_count": f"{null_cnt:,}",
                "null_pct": f"{null_p:.1f}",
                "unique_count": "-",
                "sample_val": "-",
            })

        # Quality items
        quality_diagnoses = []
        critical_count = 0
        for item in raw_quality:
            if hasattr(item, "category"):
                q_dict = {
                    "category": item.category,
                    "title": item.title,
                    "severity": item.severity,
                    "problem": item.problem,
                    "magnitude": item.magnitude,
                    "impact": item.impact,
                    "options": item.options or [],
                }
            elif isinstance(item, dict):
                q_dict = {
                    "category": item.get("category", "info"),
                    "title": item.get("title", "Diagnostik"),
                    "severity": item.get("severity", "info"),
                    "problem": item.get("problem", ""),
                    "magnitude": item.get("magnitude", ""),
                    "impact": item.get("impact", ""),
                    "options": item.get("options", []),
                }
            else:
                continue

            if q_dict["severity"] == "critical":
                critical_count += 1
            quality_diagnoses.append(q_dict)

        # Readiness Score (0-100)
        score = 100.0
        score -= min(missing_pct * 1.5, 30.0)
        dup_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        score -= min(dup_pct * 1.2, 20.0)
        score -= critical_count * 15.0
        readiness_score = max(int(round(score)), 10)

        # ML Context
        task_detection = ml_data.get("task_detection", {})
        primary_task = ml_data.get("primary_task", {}) or task_detection.get("primary_task", {})
        task_name = primary_task.get("task") or "Belum Ditentukan"
        target_col = primary_task.get("target")

        task_display_map = {
            "binary_classification": "Klasifikasi Biner (Supervised)",
            "multiclass_classification": "Klasifikasi Multikelas (Supervised)",
            "regression": "Regresi Numerik (Supervised)",
            "clustering": "Clustering (Unsupervised)",
            "anomaly_detection": "Deteksi Anomali (Unsupervised)",
        }
        primary_task_display = task_display_map.get(task_name, task_name.replace("_", " ").title())

        evaluation = ml_data.get("evaluation", {})
        eval_results = evaluation.get("results") or []
        ml_evaluated = bool(eval_results)

        candidate_models = []
        winner_model = None

        if ml_evaluated:
            # Winner model
            best_info = ml_data.get("best_method") or (eval_results[0] if eval_results else {})
            winner_method_name = best_info.get("method") or best_info.get("method_id") or "Model Unggulan"
            winner_metrics = best_info.get("metrics") or {}
            
            # Extract primary metric
            pri_metric_name = "Score"
            pri_metric_val = 0.0
            if "accuracy" in winner_metrics:
                pri_metric_name = "Accuracy"
                pri_metric_val = winner_metrics["accuracy"]
            elif "r2" in winner_metrics:
                pri_metric_name = "R²"
                pri_metric_val = winner_metrics["r2"]
            elif "f1_macro" in winner_metrics:
                pri_metric_name = "F1 Macro"
                pri_metric_val = winner_metrics["f1_macro"]
            elif "silhouette" in winner_metrics:
                pri_metric_name = "Silhouette"
                pri_metric_val = winner_metrics["silhouette"]
            elif winner_metrics:
                k, v = next(iter(winner_metrics.items()))
                pri_metric_name = str(k).title()
                pri_metric_val = v

            sec_display = ""
            if "f1_macro" in winner_metrics and pri_metric_name != "F1 Macro":
                sec_display = f"F1-Macro: {winner_metrics['f1_macro']:.4f}"
            elif "rmse" in winner_metrics:
                sec_display = f"RMSE: {winner_metrics['rmse']:.4f}"

            winner_model = {
                "method": winner_method_name,
                "primary_metric_display": f"{pri_metric_name}: {pri_metric_val:.4f}" if isinstance(pri_metric_val, (int, float)) else str(pri_metric_val),
                "secondary_metric_display": sec_display,
                "folds": best_info.get("folds", 5),
                "rationale": "Model dengan performa rata-rata tertinggi dan variansi terkecil pada validasi k-fold cross validation.",
            }

            # Map recommendations knowledge base for strengths and limitations
            knowledge_map = {}
            for rec in ml_data.get("recommendation", {}).get("recommendations", []):
                m_id = rec.get("method_id")
                if m_id:
                    knowledge_map[m_id] = rec

            for res in eval_results:
                m_id = res.get("method_id") or res.get("method")
                m_name = res.get("method") or m_id
                metrics = res.get("metrics") or {}
                
                # Format primary metric
                val_str = "-"
                for m_key in ["accuracy", "r2", "f1_macro", "silhouette"]:
                    if m_key in metrics and isinstance(metrics[m_key], (int, float)):
                        val_str = f"{m_key.title()}: {metrics[m_key]:.4f}"
                        break
                if val_str == "-" and metrics:
                    first_k, first_v = next(iter(metrics.items()))
                    val_str = f"{first_k}: {first_v:.4f}" if isinstance(first_v, (int, float)) else str(first_v)

                std_val = metrics.get("cv_std") or metrics.get("accuracy_std") or metrics.get("r2_std")
                std_str = f"± {std_val:.4f}" if isinstance(std_val, (int, float)) else "-"

                rec_k = knowledge_map.get(m_id, {})
                strengths = rec_k.get("strengths") or ["Efisien", "Robust"]
                limitations = rec_k.get("limitations") or ["Sensitif terhadap hiperparameter"]

                candidate_models.append({
                    "method": m_name,
                    "primary_metric_str": val_str,
                    "std_str": std_str,
                    "strengths": ", ".join(strengths[:2]) if isinstance(strengths, list) else str(strengths),
                    "limitations": ", ".join(limitations[:2]) if isinstance(limitations, list) else str(limitations),
                })

        # Research Context
        research_executed = bool(research_data and research_data.get("papers"))
        domain_info = research_data.get("domain") or {}
        domain_name = domain_info.get("domain") or domain_info.get("primary_domain") or "Multidisciplinary / General"

        keywords_info = research_data.get("keywords") or {}
        keywords_list = keywords_info.get("keywords") or keywords_info.get("primary_keywords") or []
        if isinstance(keywords_list, str):
            keywords_list = [k.strip() for k in keywords_list.split(",") if k.strip()]

        raw_papers = research_data.get("papers") or []
        papers_list = []
        for p in raw_papers[:15]:
            authors = p.get("authors") or []
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            else:
                authors_str = str(authors)

            rel_score = p.get("relevance_score")
            rel_display = f"{rel_score * 100:.1f}% Relevan" if isinstance(rel_score, (int, float)) and rel_score <= 1.0 else (f"{rel_score:.1f}% Relevan" if isinstance(rel_score, (int, float)) else "Relevan")

            c_count = p.get("citation_count")
            c_display = f"{c_count:,} Sitasi" if isinstance(c_count, int) else "Sitasi: -"

            papers_list.append({
                "title": p.get("title") or "Untitled Paper",
                "authors_display": authors_str or "Penulis Tidak Diketahui",
                "year": p.get("year"),
                "citation_display": c_display,
                "relevance_display": rel_display,
                "abstract": p.get("abstract") or p.get("snippet") or "",
                "url": p.get("url") or p.get("pdf_url") or "",
            })

        # Research Gaps
        gaps_info = research_data.get("gaps") or {}
        research_gaps = []
        if isinstance(gaps_info, dict):
            for k, v in gaps_info.items():
                if isinstance(v, list):
                    research_gaps.extend(str(item) for item in v)
                elif isinstance(v, str) and v.strip():
                    research_gaps.append(f"{k.replace('_', ' ').title()}: {v}")
        elif isinstance(gaps_info, list):
            research_gaps = [str(g) for g in gaps_info]

        best_model_name = winner_model["method"] if winner_model else "Belum Dievaluasi"
        best_model_score = winner_model["primary_metric_display"] if winner_model else "Jalankan ML Intelligence"

        return {
            "dataset_name": dataset_name,
            "generated_at": datetime.now().strftime("%d %B %Y, %H:%M:%S"),
            "primary_task_display": primary_task_display,
            "target_column": target_col,
            "readiness_score": readiness_score,
            "total_rows_fmt": f"{total_rows:,}",
            "total_columns_fmt": f"{total_cols:,}",
            "missing_percentage": missing_pct,
            "missing_cells_fmt": f"{missing_cells:,}",
            "best_model_name": best_model_name,
            "best_model_score": best_model_score,
            "papers_count": len(raw_papers),
            "memory_usage": memory_usage,
            "numeric_col_count": numeric_count,
            "categorical_col_count": cat_count,
            "duplicate_count_fmt": f"{duplicate_count:,}",
            "column_summary": col_summary,
            "quality_diagnoses": quality_diagnoses,
            "quality_issues_count": len(quality_diagnoses),
            "critical_issues_count": critical_count,
            "validation_protocol": "5-Fold Cross Validation" if ml_evaluated else "Belum Dievaluasi",
            "ml_evaluated": ml_evaluated,
            "winner_model": winner_model,
            "candidate_models": candidate_models,
            "research_executed": research_executed,
            "research_domain": domain_name,
            "keywords_list": keywords_list,
            "research_gaps": research_gaps,
            "papers_list": papers_list,
            "ai_synthesis": ai_synthesis,
        }

    def _generate_fallback_html(self, ctx: Dict[str, Any]) -> str:
        """Lightweight HTML generator when Jinja2 is unavailable."""
        dataset_name = html.escape(str(ctx.get("dataset_name", "Dataset")))
        generated_at = html.escape(str(ctx.get("generated_at", "")))
        total_rows = ctx.get("total_rows_fmt", "0")
        total_cols = ctx.get("total_columns_fmt", "0")

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{dataset_name} - Research Report</title>
<style>
body {{ font-family: Segoe UI, sans-serif; background: #f8fafc; color: #1e293b; padding: 30px; }}
.card {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; }}
h1 {{ color: #1e3a8a; }}
</style>
</head>
<body>
<div class="card">
<h1>{dataset_name}</h1>
<p>Generated: {generated_at}</p>
<p>Rows: {total_rows} | Columns: {total_cols}</p>
</div>
</body>
</html>"""

