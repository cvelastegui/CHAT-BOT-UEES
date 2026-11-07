from .knowledge_base import MODELS

def recommend_motorcycle(text: str):
    t = text.lower()

    if any(k in t for k in ["aventura", "montaña", "tierra", "terreno mixto", "caminos malos", "viaje", "carretera y tierra"]):
        return "XPULSE 200 4V FI", "Detecté un uso de aventura, viaje o terreno mixto."

    if any(k in t for k in ["poca gasolina", "ahorrar", "economica", "económica", "bajo consumo", "barata", "rendidora"]):
        return "ECO Deluxe", "Detecté prioridad de economía y bajo consumo."

    if any(k in t for k in ["scooter", "facil de manejar", "fácil de manejar", "automatica", "automática", "comodidad"]):
        return "XOOM 125", "Detecté prioridad de comodidad y movilidad urbana."

    if any(k in t for k in ["potencia", "deportiva", "rapida", "rápida", "160"]):
        return "HUNK 160 4V", "Detecté prioridad de desempeño."

    if any(k in t for k in ["trabajo", "delivery", "reparto", "mensajeria", "mensajería", "entregas"]):
        return "HUNK 125", "Detecté uso intensivo de trabajo urbano."

    return None, None

def rule_response(intent: str, text: str):
    model, reason = recommend_motorcycle(text)

    # Only apply recommendation rules to relevant intents
    if model and intent in {"buscar_moto","moto_economica","moto_trabajo","moto_ciudad","moto_aventura"}:
        return f"Te recomiendo **{model}**. {MODELS[model]['descripcion']} {reason}"

    if intent == "consultar_credito":
        return (
            "Podemos hacer una preorientación de crédito. En una implementación real, "
            "la solicitud se derivaría a un proceso formal de evaluación."
        )

    if intent == "consultar_modelos":
        return "Los modelos considerados en esta demo son: " + ", ".join(MODELS.keys()) + "."

    if intent == "comparar_modelos":
        return (
            "Puedo compararlos por uso, economía, potencia y terreno. "
            "ECO Deluxe prioriza economía; HUNK 160 4V desempeño; "
            "XPULSE 200 4V FI aventura y versatilidad."
        )

    return None
