from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

from model_logic import (
    analyze_topics,
    build_recommendation_text,
    get_obp_band_group as get_band_group,  # OBP band
)

app = FastAPI()

# ==== Load model once at startup ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "exam_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise e

# ==== Request body schema ====
class AttemptFeatures(BaseModel):
    student_id: int
    exam_id: int
    # topic_scores: raw scores in 0–1 range per topic
    topic_scores: dict  # e.g. { "Prof Ed - Assessment": 0.4, "Gen Ed - English": 0.7, ... }

@app.get("/")
def root():
    return {"status": "ok", "message": "LET-PRO model API is running"}

@app.post("/predict")
def predict_attempt(data: AttemptFeatures):
    # --- Clean/normalize topic scores ---
    topic_scores = {str(k): float(v) for k, v in data.topic_scores.items()}

    # --- Build feature vector ---
    try:
        feature_names = list(model.feature_names_in_)
        feature_vector = [topic_scores.get(name, 0.0) for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    except AttributeError:
        feature_names = sorted(topic_scores.keys())
        feature_vector = [topic_scores[name] for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # --- Predict band + probabilities ---
    pred_band = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0]
    class_proba = {cls: p for cls, p in zip(model.classes_, proba)}

    # --- Apply rule-based OBP band override ---
    avg_score = sum(topic_scores.values()) / len(topic_scores) if topic_scores else 0

    if avg_score >= 0.75:
        pred_band = "Exemplary"
    elif avg_score >= 0.60:
        pred_band = "Proficient"
    elif avg_score >= 0.50:
        pred_band = "Intermediate"
    else:
        pred_band = "Basic"

    # --- Confidence summaries ---
    confidence_weak_group = class_proba.get("Basic", 0.0)
    confidence_developing_group = class_proba.get("Intermediate", 0.0)
    confidence_strong_group = class_proba.get("Proficient", 0.0) + class_proba.get("Exemplary", 0.0)

    band_group = get_band_group(pred_band)

    # --- Analyze topics ---
    sorted_topics, weak_topics, dev_topics, strong_topics, all_improvement_topics = analyze_topics(topic_scores)

    recommendation_text = build_recommendation_text(pred_band)

    # --- Prepare topic table ---
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
        "predicted_band": pred_band,             
        "band_group": band_group,                
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
