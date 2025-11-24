@app.post("/predict")
def predict_attempt(data: AttemptFeatures):

    # --- Clean topic scores ---
    topic_scores = {str(k): float(v) for k, v in data.topic_scores.items()}

    # --- Build DataFrame for model probability prediction ---
    try:
        feature_names = list(model.feature_names_in_)
        feature_vector = [topic_scores.get(name, 0.0) for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    except AttributeError:
        feature_names = sorted(topic_scores.keys())
        feature_vector = [topic_scores[name] for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # --- Model probabilities (still needed for OBP confidence) ---
    proba = model.predict_proba(feature_df)[0]
    model_classes = list(model.classes_)
    class_proba = {cls: p for cls, p in zip(model_classes, proba)}

    # --- Compute exam average score and convert to OBP band ---
    avg_score = sum(topic_scores.values()) / len(topic_scores) if topic_scores else 0.0

    # New OBP Band Logic
    if avg_score < 0.50:
        obp_band = "B"   # Basic
    elif avg_score < 0.75:
        obp_band = "I"   # Intermediate
    elif avg_score < 0.90:
        obp_band = "P"   # Proficient
    else:
        obp_band = "E"   # Exemplary

    # --- Convert model probabilities to OBP groups ---
    # Here we map model classes to OBP equivalents
    obp_confidence = {"B": 0.0, "I": 0.0, "P": 0.0, "E": 0.0}

    def map_to_obp(cls):
        # Mapping based on your definition
        if cls in ["B", "D"]:
            return "B"
        elif cls == "AP":
            return "I"
        elif cls == "P":
            return "P"
        elif cls == "A":
            return "E"
        return None

    for cls, p in class_proba.items():
        mapped = map_to_obp(cls)
        if mapped:
            obp_confidence[mapped] += float(p)

    # --- Topic classification ---
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1])
    weak_topics = [t for t, s in sorted_topics if s < 0.50]
    dev_topics = [t for t, s in sorted_topics if 0.50 <= s < 0.75]
    strong_topics = [t for t, s in sorted_topics if s >= 0.75]

    improvement_topics = weak_topics + dev_topics

    # --- Recommendation text specific to OBP ---
    if obp_band == "B":
        recommendation = (
            "Your performance level is BASIC. You show partial understanding and need guided remediation. "
            "Focus strongly on the identified weak and developing topics."
        )
    elif obp_band == "I":
        recommendation = (
            "Your performance level is INTERMEDIATE. You show adequate understanding. "
            "Continue strengthening topics below proficiency for better consistency."
        )
    elif obp_band == "P":
        recommendation = (
            "Your performance level is PROFICIENT. You demonstrate solid understanding. "
            "Refine minor gaps to maintain strong performance."
        )
    elif obp_band == "E":
        recommendation = (
            "Your performance level is EXEMPLARY. Excellent mastery! "
            "Continue reinforcing all topics to sustain exceptional performance."
        )

    # --- Topic table ---
    topic_score_table = [
        {
            "topic": topic,
            "score_raw": float(score),
            "score_percent": round(score * 100, 2),
            "status": (
                "Weak" if score < 0.50
                else "Developing" if score < 0.75
                else "Strong"
            )
        }
        for topic, score in sorted_topics
    ]

    # --- Final JSON output (OBP only) ---
    return {
        "student_id": data.student_id,
        "exam_id": data.exam_id,

        # MAIN output
        "predicted_band": obp_band,  # B/I/P/E only

        # Confidence (from model probabilities mapped to OBP)
        "confidence_basic": obp_confidence["B"],
        "confidence_intermediate": obp_confidence["I"],
        "confidence_proficient": obp_confidence["P"],
        "confidence_exemplary": obp_confidence["E"],

        # Topic-level insights
        "improvement_topics": improvement_topics,
        "improvement_count": len(improvement_topics),
        "strong_topics": strong_topics,
        "strong_count": len(strong_topics),

        # Text
        "recommendation_text": recommendation,

        # Detailed topic table
        "topic_score_table": topic_score_table
    }
