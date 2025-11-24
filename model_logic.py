# --- CONFIGURATION (Based on OBP Criteria) --- 
WEAK_TOPIC_THRESHOLD = 0.50
DEVELOPING_TOPIC_THRESHOLD = 0.75

# OBP Band Mapping groups
BAND_WEAK_GROUP = ["Basic"]
BAND_DEVELOPING_GROUP = ["Intermediate"]
BAND_STRONG_GROUP = ["Proficient", "Exemplary"]

# Human-readable labels
bandLabels = {
    "Basic":        "Basic",
    "Intermediate": "Intermediate",
    "Proficient":   "Proficient",
    "Exemplary":    "Exemplary"
}

def get_obp_band_group(pred_band: str) -> str:
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
        sorted_topics: list of (topic, score) sorted ascending
        weak_topics:   topics < 0.50
        dev_topics:    topics 0.50 <= score < 0.75
        strong_topics: topics >= 0.75
        all_improvement_topics: weak + dev topics
    """
    sorted_topics = sorted(topic_scores.items(), key=lambda item: item[1])
    weak_topics = [t for t, s in sorted_topics if s < WEAK_TOPIC_THRESHOLD]
    dev_topics = [t for t, s in sorted_topics if WEAK_TOPIC_THRESHOLD <= s < DEVELOPING_TOPIC_THRESHOLD]
    strong_topics = [t for t, s in sorted_topics if s >= DEVELOPING_TOPIC_THRESHOLD]
    all_improvement_topics = weak_topics + dev_topics
    return sorted_topics, weak_topics, dev_topics, strong_topics, all_improvement_topics

def build_recommendation_text(pred_band: str) -> str:
    band_text = bandLabels.get(pred_band, pred_band)
    if pred_band == "Basic":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "Your foundation requires strong remediation. Focus immediately on the identified weak and developing topics to rebuild your core understanding."
        )
    elif pred_band == "Intermediate":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "You are developing consistency. Prioritize the identified topics to strengthen your fundamentals and avoid recurring errors."
        )
    elif pred_band == "Proficient":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "You are performing well. Reinforce the remaining areas below the Proficient threshold to maintain and stabilize your performance."
        )
    elif pred_band == "Exemplary":
        base_message = (
            f"Your projected proficiency band is {band_text}. "
            "Excellent performance. Strengthen minor weak spots for mastery and long-term retention."
        )
    else:
        base_message = (
            "Analysis complete, but the predicted band was ambiguous. "
            "Please review your topic scores and overall performance."
        )
    return base_message
