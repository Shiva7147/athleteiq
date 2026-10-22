"""
ml/explainability.py
SHAP TreeExplainer feature sensitivity attribution module.

Scientific Epistemology Note:
TreeSHAP computes exact cooperative game-theoretic Shapley attributions for the 
mathematical decision tree model. It does NOT establish biological cause-and-effect 
or clinical pathology in the human musculoskeletal system. Output descriptions reflect 
how the statistical model shifted its hazard output, not direct physiological causation.
"""
from typing import List, Dict, Any
import numpy as np
import shap
from ml.models import FEATURE_COLUMNS, MonotonicLightGBM


FEATURE_FRIENDLY_NAMES = {
    "acute_load_7d_ewma": "Acute 7-Day Workload Spike",
    "chronic_load_28d_ewma": "Chronic Workload Base",
    "acwr_ewma": "Acute:Chronic Ratio (ACWR)",
    "training_monotony_7d": "Training Monotony",
    "training_strain_7d": "Weekly Training Strain",
    "hsr_ratio_7d": "High-Speed Running Proportion",
    "sprint_exposure_7d": "Sprint Distance Exposure",
    "days_since_last_match": "Days Since Last Match",
    "mean_sleep_7d": "Self-Reported Sleep Duration",
    "fatigue_index_7d": "Self-Reported Fatigue Index",
}


class ModelExplainabilityEngine:
    """
    Computes local SHAP values for individual model outputs to power UI sensitivity inspection.
    """
    def __init__(self, monotonic_lgb: MonotonicLightGBM):
        self.tree_model = monotonic_lgb.base_estimator
        self.explainer = shap.TreeExplainer(self.tree_model)

    def explain_athlete(self, feature_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Explains top features that shifted model probability relative to base expectation.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        shap_values = self.explainer.shap_values(feature_vector)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        sorted_indices = np.argsort(sv)[::-1]
        top_drivers: List[Dict[str, Any]] = []

        for idx in sorted_indices[:top_k]:
            contrib = float(sv[idx])
            feat_name = FEATURE_COLUMNS[idx]
            raw_val = float(feature_vector[0, idx])
            friendly = FEATURE_FRIENDLY_NAMES.get(feat_name, feat_name)

            if feat_name == "sprint_exposure_7d" and contrib > 0 and raw_val < 100.0:
                summary = f"Sprint distance ({raw_val:.0f}m) lower than squad match average"
            elif feat_name == "mean_sleep_7d" and raw_val < 6.5:
                summary = f"Low self-reported sleep ({raw_val:.1f}h) increased model risk attribution"
            elif feat_name == "training_strain_7d" and contrib > 0:
                summary = f"Weekly cumulative training strain elevated ({raw_val:.0f})"
            elif feat_name == "acwr_ewma" and raw_val > 1.4:
                summary = f"Acute workload spike relative to chronic base (ACWR: {raw_val:.2f})"
            else:
                summary = f"Input {friendly} (val: {raw_val:.2f}) increased model score"

            top_drivers.append({
                "feature": feat_name,
                "label": friendly,
                "value": round(raw_val, 2),
                "shap_impact": round(contrib, 4),
                "shap_contribution": round(contrib, 4),
                "tactical_summary": summary,
                "summary": summary,
                "attribution_note": "Model feature sensitivity attribution (not biological proof)"
            })

        return top_drivers
