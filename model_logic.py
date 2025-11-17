# model_logic.py

# --- CONFIGURATION (Based on Official MPS Criteria) ---
# Weak Topics: Score < 50% (Low Proficient / Not Proficient)
WEAK_TOPIC_THRESHOLD = 0.50
# Developing Topics: 50% <= Score < 75% (Nearly Proficient)
DEVELOPING_TOPIC_THRESHOLD = 0.75
# Strong Topics: Score >= 75% (Proficient / Highly Proficient)

# DepEd Band Mapping groups (still used for confidence_Weak/Developing/Strong)
BAND_WEAK_GROUP = ["B", "D"]           # Beginning, Developing
BAND_DEVELOPING_GROUP = ["AP"]         # Approaching Proficiency
BAND_STRONG_GROUP = ["P", "A"]         # Proficient, Advanced

# Human-readable labels for the 5 bands
bandLabels = {
    "B":  "Beginning",
    "D":  "Developing",
    "AP": "Approaching Proficiency",
    "P":  "Proficient",
    "A":  "Advanced"
}


def get_deped_band_group(pred_band: str) -> str:
    """
    Maps the 5 DepEd bands (B, D, AP, P, A) 
    to 3 prescriptive groups (Weak, Developing, Strong).
    """
    if pred_band in BAND_WEAK_GROUP:
        return "Weak"
    elif pred_band in BAND_DEVELOPING_GROUP:
        return "Developing"
    elif pred_band in BAND_STRONG_GROUP:
        return "Strong"
    return "Unknown"


def analyze_topics(topic_scores: dict):
    """
    topic_scores: dict like {"Topic Name": score_in_0_to_1, ...}

    Returns:
      sorted_topics: list of (topic, score) sorted by score ascending
      weak_topics:   list of topic names with score < 0.50
      dev_topics:    list of topic names with 0.50 <= score < 0.75
      strong_topics: list of topic names with score >= 0.75
      all_improvement_topics: weak + developing topics (score < 0.75)
    """
    # Sort topics by score (lowest first) for prioritization
    sorted_topics = sorted(topic_scores.items(), key=lambda item: item[1])

    weak_topics = [
        t for t, s in sorted_topics
        if s < WEAK_TOPIC_THRESHOLD
    ]
    dev_topics = [
        t for t, s in sorted_topics
        if WEAK_TOPIC_THRESHOLD <= s < DEVELOPING_TOPIC_THRESHOLD
    ]
    strong_topics = [
        t for t, s in sorted_topics
        if s >= DEVELOPING_TOPIC_THRESHOLD
    ]

    all_improvement_topics = weak_topics + dev_topics

    return sorted_topics, weak_topics, dev_topics, strong_topics, all_improvement_topics


def build_recommendation_text(pred_band: str) -> str:
    """
    Builds the base recommendation message based on the EXACT predicted band.
    We don't attach the topic list here because topics will be sent separately
    (improvement_topics, strong_topics).
    """
    band_text = bandLabels.get(pred_band, pred_band)

    if pred_band == "B":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "Your foundation requires strong remediation. Focus immediately on the identified weak and developing topics to rebuild your core understanding"
        )
    elif pred_band == "D":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "You are developing consistency. Prioritize the identified topics to strengthen your fundamentals and avoid recurring errors"
        )
    elif pred_band == "AP":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "You are close to proficiency. Focus on topics below the 75% threshold to push your overall performance into the proficient range"
        )
    elif pred_band == "P":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "You are performing well. Reinforce the remaining areas below the Proficient threshold to maintain and stabilize your performance"
        )
    elif pred_band == "A":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "Excellent performance. Strengthen minor weak spots for mastery and long-term retention"
        )
    else:
        base_message = (
            "Analysis complete, but the predicted band was ambiguous. "
            "Please review your topic scores and overall performance."
        )

    return base_message
