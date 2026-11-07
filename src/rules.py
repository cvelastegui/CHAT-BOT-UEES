from .knowledge_base import MODELS

def recommend_motorcycle(text: str):
    t = text.lower()

    if any(k in t for k in ["aventura", "montaña", "tierra", "terreno mixto", "caminos malos", "viajes largos", "carretera y tierra"]):
        return "XPULSE 200 4V FI", "La regla detectó un uso de aventura o terreno mixto."

    if any(k in t for k in ["poca gasolina", "ahorrar combustible", "economica", "económica", "bajo consumo", "barata"]):
        return "ECO Deluxe", "La regla detectó prioridad de economía y bajo consumo."

    if any(k in t for k in ["scooter", "facil de manejar", "fácil de manejar", "automatica", "automática", "comodidad"]):
        return "XOOM 125", "La regla detectó prioridad de comodidad y movilidad urbana."

    if any(k in t for k in ["potencia", "deportiva", "rapida", "rápida", "160"]):
        return "HUNK 160 4V", "La regla detectó prioridad de desempeño."

    if any(k in t for k in ["trabajo", "delivery", "reparto", "mensajeria", "mensajería"]):
        return "HUNK 125", "La regla detectó uso intensivo de trabajo urbano."

    return None, None

def rule_response(intent: str, text: str):
    model, reason = recommend_motorcycle(text)

    if model:
        desc = MODELS[model]["descripcion"]
        return f"Te recomiendo **{model}**. {desc} {reason}"

    if intent == "consultar_credito":
        return (
            "Podemos hacer una preorientación de crédito. En una implementación real "
            "se solicitarían datos mínimos autorizados y luego se derivaría la solicitud "
            "a un proceso formal de evaluación."
        )

    if intent == "consultar_modelos":
        names = ", ".join(MODELS.keys())
        return f"Los modelos considerados en esta demo son: {names}."

    if intent == "comparar_modelos":
        return (
            "Puedo compararlos por uso, economía, potencia y tipo de terreno. "
            "Por ejemplo, ECO Deluxe prioriza economía; HUNK 160 4V prioriza desempeño; "
            "XPULSE 200 4V FI prioriza aventura y versatilidad."
        )

    return None
