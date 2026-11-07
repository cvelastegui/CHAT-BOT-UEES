from src.chatbot import get_bot_response

TESTS = [
    ("Hola", "saludo"),
    ("Buenos días", "saludo"),
    ("Quiero comprar una moto", "buscar_moto"),
    ("Busco una moto económica", "moto_economica"),
    ("Necesito una moto para trabajar", "moto_trabajo"),
    ("La usaré en ciudad", "moto_ciudad"),
    ("Quiero una moto para montaña", "moto_aventura"),
    ("Cuánto cuesta la HUNK 125", "consultar_precio"),
    ("Tienen financiamiento", "consultar_credito"),
    ("Qué modelos tienen", "consultar_modelos"),
    ("Quiero comparar la ECO y la HUNK", "comparar_modelos"),
    ("Quiero hablar con un asesor", "contacto_asesor"),
    ("Gracias por la ayuda", "despedida"),
    ("Quiero una moto que gaste poco combustible", "moto_economica"),
    ("La necesito para delivery", "moto_trabajo"),
    ("Busco una moto para carretera y tierra", "moto_aventura"),
    ("Puedo pagar mensualmente", "consultar_credito"),
    ("Dime qué motocicletas HERO existen", "consultar_modelos"),
    ("Cuál me conviene más entre dos modelos", "comparar_modelos"),
    ("Hasta luego", "despedida")
]

if __name__ == "__main__":
    correct = 0
    for i, (text, expected) in enumerate(TESTS, start=1):
        result = get_bot_response(text)
        pred = result["intent"]
        passed = pred == expected
        correct += int(passed)
        print(
            f"{i:02d}. {'OK' if passed else 'FALLO'} | "
            f"{text} | esperado={expected} | predicho={pred} | "
            f"confianza={result['confidence']:.1%}"
        )

    print(f"\nResultado: {correct}/{len(TESTS)}")
    print(f"Tasa de éxito: {correct/len(TESTS):.1%}")
