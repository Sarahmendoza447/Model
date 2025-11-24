# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import joblib
import logging
import pandas as pd

# ------- logging -------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------- FastAPI app (MUST be defined before route decorators) -------
app = FastAPI()

# ------- global model holder -------
model = None
MODEL_FILENAME = "exam_model.pkl"

# ------- request schema -------
class AttemptFeatures(BaseModel):
    student_id: int
    exam_id: int
    topic_scores: Dict[str, float]

# ------- startup: load model -------
@app.on_event("startup")
def load_model():
    global model
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, MODEL_FILENAME)
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")
    except Exception as e:
        model = None
        logger.exception(f"Failed to load model: {e}")

# ------- health endpoint -------
@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

# ------- helper: map legacy model class -> OBP -------
def map_model_class_to_obp(cls: str):
    """
    If your model's classes are B/D/AP/P/A, map them to OBP:
      B,D -> B (Basic)
      AP   -> I (Intermediate)
      P    -> P (Proficient)
      A    -> E (Exemplary)
    """
    if cls in ("B", "D"):
        return "B"
    if cls == "AP":
        return "I"
    if cls == "P":
        return "P"
    if cls == "A":
        return "E"
    return None

# ------- main predict endpoint (OBP-only) -------
@app.post("/predict")
def predict_attempt(data: AttemptFeatures):
    # Ensure model loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server. Check logs.")

    # Normalize topic scores
    topic_scores = {str(k): float(v) for k, v in data.topic_scores.items()}

    # Build feature dataframe in same order as training
    try:
        feature_names = list(model.feature_names_in_)
        feature_vector = [topic_scores.get(name, 0.0) for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    except Exception:
        feature_names = sorted(topic_scores.keys())
        feature_vector = [topic_scores[name] for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # Compute average score and convert to OBP band (score-based)
    avg_score = sum(topic_scores.values()) / len(topic_scores) if topic_scores else 0.0
    if avg_score < 0.50:
        obp_band = "B"   # Basic
    elif avg_score < 0.75:
        obp_band = "I"   # Intermediate
    elif avg_score < 0.90:
        obp_band = "P"   # Proficient
    else:
        obp_band = "E"   # Exemplary

    # Get model probabilities (for confidence) if available
    class_proba = {}
    try:
        proba = model.predict_proba(feature_df)[0]
        model_classes = list(model.classes_)
        class_proba = {cls: float(p) for cls, p in zip(model_classes, proba)}
    except Exception:
        logger.info("Model predict_proba not available or failed; proceeding without class probabilities.")

    # Map model class probabilities into OBP confidence bins
    obp_confidence = {"B": 0.0, "I": 0.0, "P": 0.0, "E": 0.0}
    for cls, p in class_proba.items():
        mapped = map_model_class_to_obp(cls)
        if mapped:
            obp_confidence[mapped] += p

    # Topic analysis (same thresholds)
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1])
    weak_topics = [t for t, s in sorted_topics if s < 0.50]
    dev_topics = [t for t, s in sorted_topics if 0.50 <= s < 0.75]
    strong_topics = [t for t, s in sorted_topics if s >= 0.75]
    improvement_topics = weak_topics + dev_topics

    # Recommendation text (OBP)
    if obp_band == "B":
        recommendation = "Your performance level is BASIC. Partial understanding — focus on weak topics with guidance."
    elif obp_band == "I":
        recommendation = "Your performance level is INTERMEDIATE. Adequate understanding — strengthen developing topics."
    elif obp_band == "P":
        recommendation = "Your performance level is PROFICIENT. Strong understanding — refine minor gaps."
    else:
        recommendation = "Your performance level is EXEMPLARY. Exceptional mastery — continue challenging work."

    # Prepare topic table
    topic_score_table = [
        {
            "topic": topic,
            "score_raw": float(score),
            "score_percent": round(score * 100, 2),
            "status": ("Weak" if score < 0.50 else "Developing" if score < 0.75 else "Strong")
        }
        for topic, score in sorted_topics
    ]

    return {
        "student_id": data.student_id,
        "exam_id": data.exam_id,
        "predicted_band": obp_band,                    # B/I/P/E only
        "predicted_band_label": {"B":"Basic","I":"Intermediate","P":"Proficient","E":"Exemplary"}.get(obp_band, obp_band),
        "confidence_basic": obp_confidence["B"],
        "confidence_intermediate": obp_confidence["I"],
        "confidence_proficient": obp_confidence["P"],
        "confidence_exemplary": obp_confidence["E"],
        "improvement_topics": improvement_topics,
        "improvement_count": len(improvement_topics),
        "strong_topics": strong_topics,
        "strong_count": len(strong_topics),
        "recommendation_text": recommendation,
        "topic_score_table": topic_score_table
    }
