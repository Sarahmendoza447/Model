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
        feature_names = list(model.feature_names_in_)
        feature_vector = [topic_scores.get(name, 0.0) for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    except AttributeError:
        feature_names = sorted(topic_scores.keys())
        feature_vector = [topic_scores[name] for name in feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # --- Predict using model only (NO OVERRIDE) ---
    pred_band = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0]
    class_proba = {cls: p for cls, p in zip(model.classes_, proba)}

    # Confidence grouping
    confidence_weak_group = class_proba.get("B", 0.0) + class_proba.get("D", 0.0)
    confidence_developing_group = class_proba.get("AP", 0.0)
    confidence_strong_group = class_proba.get("P", 0.0) + class_proba.get("A", 0.0)

    band_group = get_deped_band_group(pred_band)

    # --- Topic analysis ---
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
