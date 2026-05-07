# =========================
# 🎭 PERSONA CONFIGS
# =========================

PERSONAS = {
    "Analyst": {
        "icon": "📊",
        "tone": "technical",
        "summary_style": "detailed",
        "focus": [
            "patterns",
            "statistics",
            "correlations",
            "data quality"
        ]
    },

    "CEO": {
        "icon": "🧑‍💼",
        "tone": "strategic",
        "summary_style": "executive",
        "focus": [
            "growth",
            "risk",
            "opportunities",
            "performance"
        ]
    },

    "Marketing": {
        "icon": "📣",
        "tone": "growth-focused",
        "summary_style": "engaging",
        "focus": [
            "customers",
            "engagement",
            "demand",
            "conversion"
        ]
    }
}


# =========================
# 🎨 UI COLORS
# =========================

COLORS = {
    "success": "#16a34a",
    "warning": "#f59e0b",
    "danger": "#dc2626",
    "primary": "#2563eb"
}


# =========================
# 📊 DATASET HEALTH SETTINGS
# =========================

HEALTH_RULES = {
    "missing_threshold": 20,
    "duplicate_threshold": 10,
    "small_dataset_rows": 50
}


# =========================
# 🤖 AI SETTINGS
# =========================

AI_SETTINGS = {
    "temperature": 0.7,
    "max_history": 5
}