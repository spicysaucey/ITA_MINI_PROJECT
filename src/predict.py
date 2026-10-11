"""Prediction pipeline for Cyberbullying and Harassment Detection.

Provides single-comment and high-throughput batch classification with:
- Binary verdict ('Attacking' vs 'Not Attacking')
- Calibrated confidence percentage (%)
- Multi-class harassment type taxonomy (Ethnicity, Political, Religion, Sexual, Threat, Troll, Vocational)
- Severity level mapping (High, Medium, Low, None)
- Cached model loader for optimal latency
"""

import os
import sys

# Ensure root directory is on sys.path for relative imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

try:
    from src.preprocess import clean_text
except ImportError:
    from preprocess import clean_text

# Default model file paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "binary_lr.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "binary_svm.joblib")
TYPE_MODEL_PATH = os.path.join(MODELS_DIR, "type_classifier.joblib")

# Severity level definition based on threat level and real-world harm impact
SEVERITY_MAPPING = {
    "Threat": "High",
    "Sexual": "High",
    "Ethnicity": "High",
    "Political": "Medium",
    "Religion": "Medium",
    "Vocational": "Medium",
    "Troll": "Low"
}

# Cache for loaded models in memory
_MODEL_CACHE: Dict[str, Any] = {}


def load_model_pipelines() -> Dict[str, Any]:
    """Load trained model pipelines into memory. Cached across function calls."""
    global _MODEL_CACHE
    if not _MODEL_CACHE:
        if not os.path.exists(LR_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {LR_MODEL_PATH}")
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {SVM_MODEL_PATH}")
        if not os.path.exists(TYPE_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {TYPE_MODEL_PATH}")

        _MODEL_CACHE["lr"] = joblib.load(LR_MODEL_PATH)
        _MODEL_CACHE["svm"] = joblib.load(SVM_MODEL_PATH)
        _MODEL_CACHE["type"] = joblib.load(TYPE_MODEL_PATH)

    return _MODEL_CACHE


def predict_comment(
    raw_text: str,
    model_name: str = "Logistic Regression"
) -> Dict[str, Any]:
    """Run full inference on a single raw comment string.

    Args:
        raw_text: Raw comment text.
        model_name: Either 'Logistic Regression' or 'Linear SVM'.

    Returns:
        Dict containing prediction details and confidence scores.
    """
    cleaned = clean_text(raw_text)

    # Handle empty or whitespace text
    if not cleaned:
        return {
            "raw_text": raw_text or "",
            "cleaned_text": "",
            "verdict": "Not Attacking",
            "is_attacking": False,
            "confidence_score": 0.50,
            "confidence_pct": 50.0,
            "attack_type": None,
            "type_confidence": 0.0,
            "severity": "None",
            "model_used": model_name
        }

    pipelines = load_model_pipelines()
    selected_pipeline = pipelines["lr"] if "logistic" in model_name.lower() else pipelines["svm"]
    type_pipeline = pipelines["type"]

    # 1. Binary Classification
    probs = selected_pipeline.predict_proba([cleaned])[0]
    classes = list(selected_pipeline.classes_)
    attacking_idx = classes.index("Attacking")
    prob_attacking = float(probs[attacking_idx])
    prob_not_attacking = 1.0 - prob_attacking

    is_attacking = prob_attacking >= 0.50
    verdict = "Attacking" if is_attacking else "Not Attacking"
    confidence_score = prob_attacking if is_attacking else prob_not_attacking
    confidence_pct = round(confidence_score * 100.0, 1)

    # 2. Harassment Type Classification (only for flagged attacking comments)
    attack_type: Optional[str] = None
    type_confidence: float = 0.0
    severity: str = "None"

    if is_attacking:
        type_probs = type_pipeline.predict_proba([cleaned])[0]
        type_classes = list(type_pipeline.classes_)
        top_type_idx = int(np.argmax(type_probs))
        attack_type = type_classes[top_type_idx]
        type_confidence = round(float(type_probs[top_type_idx]) * 100.0, 1)
        severity = SEVERITY_MAPPING.get(attack_type, "Medium")

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "verdict": verdict,
        "is_attacking": is_attacking,
        "confidence_score": round(confidence_score, 4),
        "confidence_pct": confidence_pct,
        "attack_type": attack_type,
        "type_confidence": type_confidence,
        "severity": severity,
        "model_used": model_name
    }


def predict_batch(
    comments: List[str],
    model_name: str = "Logistic Regression"
) -> pd.DataFrame:
    """Run optimized batch inference on a list of comment strings.

    Args:
        comments: List of raw comment strings.
        model_name: Either 'Logistic Regression' or 'Linear SVM'.

    Returns:
        pd.DataFrame with prediction columns and summary stats.
    """
    if not comments:
        return pd.DataFrame(columns=[
            "Comment", "Verdict", "Confidence (%)", "Harassment Type",
            "Type Confidence (%)", "Severity", "Is Attacking"
        ])

    pipelines = load_model_pipelines()
    selected_pipeline = pipelines["lr"] if "logistic" in model_name.lower() else pipelines["svm"]
    type_pipeline = pipelines["type"]

    cleaned_texts = [clean_text(c) for c in comments]

    # Replace completely empty strings with placeholder token to avoid vectorizer errors
    clean_series = [t if t else "safe" for t in cleaned_texts]

    # Batch binary inference
    probs = selected_pipeline.predict_proba(clean_series)
    classes = list(selected_pipeline.classes_)
    attacking_idx = classes.index("Attacking")
    prob_attacking_vals = probs[:, attacking_idx]

    verdicts = []
    is_attacking_flags = []
    confidence_pcts = []

    for i, pa in enumerate(prob_attacking_vals):
        if not cleaned_texts[i]:  # original empty text
            verdicts.append("Not Attacking")
            is_attacking_flags.append(False)
            confidence_pcts.append(50.0)
        elif pa >= 0.50:
            verdicts.append("Attacking")
            is_attacking_flags.append(True)
            confidence_pcts.append(round(pa * 100.0, 1))
        else:
            verdicts.append("Not Attacking")
            is_attacking_flags.append(False)
            confidence_pcts.append(round((1.0 - pa) * 100.0, 1))

    # Harassment type inference for attacking items
    flagged_indices = [idx for idx, flag in enumerate(is_attacking_flags) if flag]
    types_list: List[Optional[str]] = [None] * len(comments)
    type_conf_list: List[float] = [0.0] * len(comments)
    severity_list: List[str] = ["None"] * len(comments)

    if flagged_indices:
        flagged_clean = [clean_series[idx] for idx in flagged_indices]
        type_probs = type_pipeline.predict_proba(flagged_clean)
        type_classes = list(type_pipeline.classes_)
        top_indices = np.argmax(type_probs, axis=1)

        for f_pos, orig_idx in enumerate(flagged_indices):
            t_idx = top_indices[f_pos]
            t_label = type_classes[t_idx]
            t_conf = round(float(type_probs[f_pos, t_idx]) * 100.0, 1)
            types_list[orig_idx] = t_label
            type_conf_list[orig_idx] = t_conf
            severity_list[orig_idx] = SEVERITY_MAPPING.get(t_label, "Medium")

    df_results = pd.DataFrame({
        "Comment": comments,
        "Verdict": verdicts,
        "Confidence (%)": confidence_pcts,
        "Harassment Type": types_list,
        "Type Confidence (%)": type_conf_list,
        "Severity": severity_list,
        "Is Attacking": is_attacking_flags
    })

    return df_results


if __name__ == "__main__":
    sample_comments = [
        "What a fantastic explanation, thank you so much!",
        "Shut up you dumb clown go kill yourself 🤡",
        "Your religion is disgusting and backwards get out of here",
        "Does this laptop support 64GB DDR5 memory upgrade?",
    ]
    print("Testing single inference:")
    for sc in sample_comments:
        res = predict_comment(sc)
        print(f"Text: {sc[:40]}... -> Verdict: {res['verdict']} ({res['confidence_pct']}%) | Type: {res['attack_type']} | Severity: {res['severity']}")

    print("\nTesting batch inference:")
    df_res = predict_batch(sample_comments)
    print(df_res[["Verdict", "Confidence (%)", "Harassment Type", "Severity"]])
