# model_logic.py

# --- CONFIGURATION (Based on Official MPS Criteria) ---
# Weak Topics: Score < 50% (Low Proficient / Not Proficient)
WEAK_TOPIC_THRESHOLD = 0.50
# Developing Topics: 50% <= Score < 75% (Nearly Proficient)
DEVELOPING_TOPIC_THRESHOLD = 0.75
# Strong Topics: Score >= 75% (Proficient / Highly Proficient)


# ======================================================
#                   OBP (Score-based) MAPPING
# ======================================================
# OBP bands:
# B -> Basic (0.00 - 0.49)
# I -> Intermediate (0.50 - 0.74)
# P -> Proficient (0.75 - 0.89)
# E -> Exemplary (0.90 - 1.00)

bandLabels = {
    "B": "Basic (Emerging)",
    "I": "Intermediate (Developing)",
    "P": "Proficient (Advanced)",
    "E": "Exemplary (Mastery)"
}


def convert_score_to_obp(avg_score: float) -> str:
    """
    Convert a 0–1 average score to OBP band:
      avg < 0.50  -> "B" (Basic)
      0.50 <= avg < 0.75 -> "I" (Intermediate)
      0.75 <= avg < 0.90 -> "P" (Proficient)
      avg >= 0.90 -> "E" (Exemplary)
    """
    if avg_score is None:
        return "Unknown"
    try:
        s = float(avg_score)
    except Exception:
        return "Unknown"

    if s < 0.50:
        return "B"
    elif s < 0.75:
        return "I"
    elif s < 0.90:
        return "P"
    else:
        return "E"


# Compatibility: map old 5-band model outputs to OBP (if you still get model preds as B/D/AP/P/A)
MODEL_TO_OBP_MAP = {
    "B": "B",   # Beginning -> Basic
    "D": "B",   # Developing -> Basic
    "AP": "I",  # Approaching Proficiency -> Intermediate
    "P": "P",   # Proficient -> Proficient
    "A": "E"    # Advanced -> Exemplary
}


def map_model_band_to_obp(model_band: str) -> str:
    """
    Convert legacy model band (B, D, AP, P, A) to OBP (B, I, P, E).
    Returns 'Unknown' if mapping not found.
    """
    return MODEL_TO_OBP_MAP.get(model_band, "Unknown")


def get_obp_confidence_group(obp_band: str) -> str:
    """
    Return a coarse group label for confidence reporting (keeps app.py's existing
    'band_group' semantics if you want to keep 'Weak'/'Developing'/'Strong').
    Mapping:
      B -> Weak
      I -> Developing
      P, E -> Strong
    """
    if obp_band == "B":
        return "Weak"
    elif obp_band == "I":
        return "Developing"
    elif obp_band in ("P", "E"):
        return "Strong"
    return "Unknown"


# ======================================================
#                Topic analysis helpers
# ======================================================
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

    weak_topics = [t for t, s in sorted_topics if s < WEAK_TOPIC_THRESHOLD]
    dev_topics = [t for t, s in sorted_topics if WEAK_TOPIC_THRESHOLD <= s < DEVELOPING_TOPIC_THRESHOLD]
    strong_topics = [t for t, s in sorted_topics if s >= DEVELOPING_TOPIC_THRESHOLD]

    all_improvement_topics = weak_topics + dev_topics

    return sorted_topics, weak_topics, dev_topics, strong_topics, all_improvement_topics


# ======================================================
#             Recommendation / Messaging
# ======================================================
def build_recommendation_text(obp_band: str) -> str:
    """
    Build a recommendation message using the NEW OBP band (B, I, P, E).
    Supply the OBP band code (not the old DepEd 5-band).

    If you have an average score instead, call convert_score_to_obp(avg_score)
    and pass that result here.
    """
    band_text = bandLabels.get(obp_band, obp_band)

    if obp_band == "B":
        return (
            f"Your performance level is {band_text}. "
            "You demonstrate partial understanding and require guidance to apply knowledge and skills. "
            "Begin by focusing on the weakest topics first to build a stronger foundation."
        )

    if obp_band == "I":
        return (
            f"Your performance level is {band_text}. "
            "You demonstrate adequate understanding and can apply knowledge with some independence. "
            "Target developing topics below the 75% threshold to progress toward proficiency."
        )

    if obp_band == "P":
        return (
            f"Your performance level is {band_text}. "
            "You demonstrate full understanding and can apply knowledge independently and effectively. "
            "Continue reinforcing minor weaknesses to sustain performance."
        )

    if obp_band == "E":
        return (
            f"Your performance level is {band_text}. "
            "You demonstrate exceptional understanding and can extend knowledge to new contexts. "
            "Challenge yourself with more complex problems to maintain and expand mastery."
        )

    return "Performance analysis completed, but the performance level could not be determined."
