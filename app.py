# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

from model_logic import (
    analyze_topics,
    build_recommendation_text,
    get_deped_band_group,
)

app = FastAPI()

# ==== Load model once at startup ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "exam_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    # On Azure this will appear in logs
    print(f"ERROR loading model: {e}")
    raise e


# ==== Request body schema ====
class AttemptFeatures(BaseModel):
    student_id: int
    exam_id: int
    # topic_scores: raw scores in 0–1 range per topic (same as your training logic)
    topic_scores: dict  # e.g. { "Prof Ed - Assessment": 0.4, "Gen Ed - English": 0.7, ... }


@app.get("/")
def root():
    return {"status": "ok", "message": "LET-PRO model API is running"}


@app.post("/predict")
def predict_attempt(data: AttemptFeatures):
    """
    Expects JSON like:
    {
      "student_id": 123,
      "exam_id": 45,
      "topic_scores": {
        "Prof Ed - Assessment": 0.40,
        "Prof Ed - Dev of Learners": 0.55,
        "Gen Ed - English": 0.70
      }
    }
    """
    # --- Clean/normalize topic scores to float ---
    topic_scores = {str(k): float(v) for k, v in data.topic_scores.items()}

    # --- Build feature vector in the same order as training ---
    try:
        # If the model was trained with feature_names_in_, we respect that order
        feature_names = list(model.feature_names_in_)
        feature_vector = [topic_scores.get(name, 0.0) for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    except AttributeError:
        # Fallback: use sorted keys if feature_names_in_ is not available
        feature_names = sorted(topic_scores.keys())
        feature_vector = [topic_scores[name] for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # --- Predict band + probabilities ---
    pred_band = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0]
    class_proba = {cls: p for cls, p in zip(model.classes_, proba)}

    # --- Apply rule-based override for clear performance boundaries ---
    # Calculate average score across all topics
    avg_score = sum(topic_scores.values()) / len(topic_scores) if topic_scores else 0
    
    # Override prediction if performance clearly indicates a different band
    if avg_score >= 0.90:  # 90% or higher -> Advanced
        pred_band = "A"
    elif avg_score >= 0.85:  # 85-89% -> Proficient
        pred_band = "P"
    elif avg_score >= 0.80:  # 80-84% -> Approaching Proficiency
        pred_band = "AP"
    elif avg_score >= 0.75:  # 75-79% -> Developing
        pred_band = "D"
    elif avg_score < 0.75:  # Below 75% -> Beginning
        pred_band = "B"

    confidence_weak_group = class_proba.get("B", 0.0) + class_proba.get("D", 0.0)
    confidence_developing_group = class_proba.get("AP", 0.0)
    confidence_strong_group = class_proba.get("P", 0.0) + class_proba.get("A", 0.0)

    band_group = get_deped_band_group(pred_band)

    # --- Analyze topics using the official MPS boundaries (50% and 75%) ---
    sorted_topics, weak_topics, dev_topics, strong_topics, all_improvement_topics = analyze_topics(topic_scores)

    recommendation_text = build_recommendation_text(pred_band)

    # --- Prepare response ---
    topic_score_table = [
        {
            "topic": topic,
            "score_raw": float(score),
            "score_percent": round(float(score) * 100, 2),
            "status": (
                "Weak" if score < 0.50
                else "Developing" if score < 0.75
                else "Strong"
            ),
        }
        for topic, score in sorted_topics
    ]

    return {
        "student_id": data.student_id,
        "exam_id": data.exam_id,
        "predicted_band": pred_band,             # B, D, AP, P, A
        "band_group": band_group,                # Weak / Developing / Strong
        "confidence_weak": confidence_weak_group,
        "confidence_developing": confidence_developing_group,
        "confidence_strong": confidence_strong_group,
        "improvement_topics": all_improvement_topics,
        "improvement_count": len(all_improvement_topics),
        "strong_topics": strong_topics,
        "strong_count": len(strong_topics),
        "recommendation_text": recommendation_text,
        "topic_score_table": topic_score_table
    }
