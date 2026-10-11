"""Model explainability and token-level feature attribution module.

Provides:
- Linear feature attribution (TF-IDF weight * classifier coefficient)
- Identification of influential attacking vs protective tokens
- HTML span generation for highlighting influential words
- Ethical guidelines and human-in-the-loop review disclaimer
"""

import os
import sys
import re
import html
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocess import clean_text
from src.predict import load_model_pipelines

ETHICS_NOTICE = (
    "⚠️ **Important Ethics & Human-in-the-Loop Notice**: "
    "All automated classifications and highlighted terms are advisory indicators intended "
    "to assist human moderation teams. Algorithmic scores must not serve as sole arbiters of disciplinary "
    "action, as context, nuance, linguistic sarcasm, and cultural dialects may influence model output."
)


def explain_comment(
    raw_text: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Calculate token attributions and extract influential features for a comment.

    Args:
        raw_text: Original text comment.
        top_k: Maximum number of influential terms to extract for each side.

    Returns:
        Dict containing attacking_words, safe_words, highlighted_html, and ethics_notice.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {
            "attacking_words": [],
            "safe_words": [],
            "highlighted_html": html.escape(raw_text or ""),
            "ethics_notice": ETHICS_NOTICE
        }

    pipelines = load_model_pipelines()
    lr = pipelines["lr"]

    tfidf_step = lr.named_steps["tfidf"]
    clf_step = lr.named_steps["clf"]

    # Transform input text
    X_vec = tfidf_step.transform([cleaned])
    feature_names = tfidf_step.get_feature_names_out()
    coefs = clf_step.coef_[0]

    # Non-zero entries for this document
    coo = X_vec.tocoo()
    contributions = []

    for col_idx, tfidf_val in zip(coo.col, coo.data):
        feat_name = feature_names[col_idx]
        # In this model: negative coefficient = Attacking, positive = Not Attacking
        # We calculate attack_impact = - (coef * tfidf_val)
        raw_coef = coefs[col_idx]
        attack_impact = -1.0 * float(raw_coef * tfidf_val)

        # Clean display name from transformer prefix
        display_name = feat_name.replace("word_tfidf__", "").replace("char_tfidf__", "")
        # Filter out purely character-ngram noise if desired, keep readable tokens
        if not feat_name.startswith("char_tfidf__") or len(display_name.strip()) > 3:
            contributions.append({
                "feature": display_name,
                "raw_feature": feat_name,
                "impact": attack_impact,
                "tfidf": float(tfidf_val)
            })

    # Sort: highest attack_impact -> strongest attacking features
    contributions.sort(key=lambda x: x["impact"], reverse=True)

    attacking_features = [c for c in contributions if c["impact"] > 0.05][:top_k]
    safe_features = [c for c in contributions if c["impact"] < -0.05]
    safe_features.reverse()
    safe_features = safe_features[:top_k]

    # Highlight words in the raw text
    highlighted_html = generate_highlighted_html(raw_text, attacking_features, safe_features)

    return {
        "attacking_words": attacking_features,
        "safe_words": safe_features,
        "highlighted_html": highlighted_html,
        "ethics_notice": ETHICS_NOTICE
    }


def generate_highlighted_html(
    raw_text: str,
    attacking_features: List[Dict[str, Any]],
    safe_features: List[Dict[str, Any]]
) -> str:
    """Generate HTML text with styled spans for influential terms."""
    if not raw_text:
        return ""

    escaped_text = html.escape(raw_text)

    # Collect unique attacking words
    attack_words = set()
    for item in attacking_features:
        feat = item["feature"].strip()
        if feat and len(feat) > 2 and not feat.startswith("["):
            attack_words.add(feat)

    # Highlight attacking words in red/coral badge
    for w in sorted(attack_words, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<span style="background-color: rgba(239, 68, 68, 0.25); border-bottom: 2px solid #ef4444; color: #f87171; padding: 2px 4px; border-radius: 4px; font-weight: 600;" title="Influential Risk Indicator">\1</span>',
            escaped_text
        )

    # Collect safe/positive words
    safe_words = set()
    for item in safe_features:
        feat = item["feature"].strip()
        if feat and len(feat) > 2 and not feat.startswith("["):
            safe_words.add(feat)

    # Highlight safe words in emerald/green badge
    for w in sorted(safe_words, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<span style="background-color: rgba(16, 185, 129, 0.25); border-bottom: 2px solid #10b981; color: #34d399; padding: 2px 4px; border-radius: 4px; font-weight: 500;" title="Protective / Positive Feature">\1</span>',
            escaped_text
        )

    return escaped_text


if __name__ == "__main__":
    import re
    test_comment = "You are a dirty loser clown, stop talking and delete this video!"
    res = explain_comment(test_comment)
    print("Influential attacking words:")
    for w in res["attacking_words"]:
        print(f" - {w['feature']} (impact: {w['impact']:.3f})")
    print("\nHighlighted HTML preview:")
    print(res["highlighted_html"])
