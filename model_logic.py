# --- CONFIGURATION (Based on OBP Criteria) ---
# Weak Topics: Score < 50% (Basic)
WEAK_TOPIC_THRESHOLD = 0.50
# Developing Topics: 50% <= Score < 75% (Intermediate)
DEVELOPING_TOPIC_THRESHOLD = 0.75
# Strong Topics: Score >= 75% (Proficient / Exemplary)

# OBP Band Mapping groups (used for confidence_Weak/Developing/Strong)
BAND_WEAK_GROUP = ["Basic"]           # Basic
BAND_DEVELOPING_GROUP = ["Intermediate"] # Intermediate
BAND_STRONG_GROUP = ["Proficient", "Exemplary"]  # Proficient, Exemplary

# Human-readable labels for the 4 OBP bands
bandLabels = {
    "Basic":        "Basic",
    "Intermediate": "Intermediate",
    "Proficient":   "Proficient",
    "Exemplary":    "Exemplary"
}


def get_obp_band_group(pred_band: str) -> str:
    """
    Maps the 4 OBP bands to 3 prescriptive groups (Weak, Developing, Strong)
    for confidence summaries.
    """
    if pred_band in BAND_WEAK_GROUP:
        return "Weak"
    elif pred_band in BAND_DEVELOPING_GROUP:
        return "Developing"
    elif pred_band in BAND_STRONG_GROUP:
        return "Strong"
    return "Unknown"


def build_recommendation_text(pred_band: str) -> str:
    """
    Builds the base recommendation message based on the EXACT predicted OBP band.
    """
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
